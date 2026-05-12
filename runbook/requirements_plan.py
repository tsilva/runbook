from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import yaml

from runbook.files import atomic_write_text

DEFAULT_TIMEOUT = 3600
DEFAULT_KERNEL_NAME = "python3"
DEFAULT_PYTHON_VERSION = "3.11"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.5"
DEFAULT_OPENROUTER_REASONING_EFFORT = "high"
DEFAULT_OPENROUTER_MAX_COMPLETION_TOKENS = 16000
DEFAULT_OPENROUTER_RETRIES = 2
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 120
DEFAULT_OPENROUTER_MAX_NOTEBOOK_CHARS = 200_000
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
RequirementsPlanner = Callable[[str, str, str | None], dict[str, Any]]
BUILD_TOOLCHAIN_PIP_PACKAGES = {
    "bitsandbytes",
    "deepspeed",
    "flash-attn",
    "flash-attn-2",
    "flash-attn-3",
    "liger-kernel",
    "triton",
    "unsloth",
    "unsloth-zoo",
    "xformers",
}


@dataclass(frozen=True)
class RuntimeRequirements:
    image: str | None = None
    gpu: str | None = None
    cpu: float | None = None
    memory: int | None = None
    timeout: int = DEFAULT_TIMEOUT
    kernel_name: str = DEFAULT_KERNEL_NAME
    python_version: str = DEFAULT_PYTHON_VERSION
    build_toolchain: bool = True
    pip_index_url: str | None = None
    pip_extra_index_urls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PackageRequirements:
    pip: list[str] = field(default_factory=list)
    apt: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModalRequirements:
    secrets: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlannerMetadata:
    provider: str = "openrouter"
    model: str = DEFAULT_OPENROUTER_MODEL
    generated_at: str | None = None
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)
    source_hash: str | None = None


@dataclass(frozen=True)
class NotebookRequirements:
    version: int = 1
    runtime: RuntimeRequirements = field(default_factory=RuntimeRequirements)
    packages: PackageRequirements = field(default_factory=PackageRequirements)
    modal: ModalRequirements = field(default_factory=ModalRequirements)
    planner: PlannerMetadata = field(default_factory=PlannerMetadata)


@dataclass(frozen=True)
class RequirementsLoadResult:
    path: Path
    requirements: NotebookRequirements
    generated: bool
    previous_requirements: NotebookRequirements | None = None


class RequirementsConfigError(RuntimeError):
    """Raised when a companion requirements file cannot be loaded or generated."""


def companion_requirements_path(input_path: Path) -> Path:
    source_path = input_path.expanduser().resolve()
    return source_path.with_name(f"{source_path.name}.yaml")


def load_or_generate_requirements(
    input_path: Path,
    notebook_json: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    regenerate: bool = False,
    planner: RequirementsPlanner | None = None,
) -> RequirementsLoadResult:
    path = companion_requirements_path(input_path)
    if path.exists() and not regenerate:
        requirements = load_requirements(path)
        _ensure_requirements_current(requirements, notebook_json, path)
        return RequirementsLoadResult(
            path=path,
            requirements=requirements,
            generated=False,
        )

    previous_requirements = load_requirements(path) if path.exists() else None
    requirements = generate_requirements(
        notebook_json,
        model=model,
        api_key=api_key,
        planner=planner,
    )
    write_requirements(path, requirements)
    return RequirementsLoadResult(
        path=path,
        requirements=requirements,
        generated=True,
        previous_requirements=previous_requirements,
    )


def load_requirements(path: Path) -> NotebookRequirements:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RequirementsConfigError(f"Could not read requirements file {path}: {exc}") from exc
    return parse_requirements(raw, source=str(path))


def write_requirements(path: Path, requirements: NotebookRequirements) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(requirements_to_dict(requirements), sort_keys=False),
    )


def generate_requirements(
    notebook_json: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    planner: RequirementsPlanner | None = None,
) -> NotebookRequirements:
    selected_model = model or os.environ.get("RUNBOOK_OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    notebook_text = notebook_json_to_jupytext(notebook_json)
    notebook_text = _prepare_notebook_text_for_planner(notebook_text)
    planner_func = planner or _plan_requirements_with_openrouter
    raw_plan = planner_func(notebook_text, selected_model, api_key)
    requirements = parse_requirements(raw_plan, source="OpenRouter response")
    return _with_planner_defaults(
        requirements,
        selected_model,
        source_hash=notebook_source_hash(notebook_json),
    )


def notebook_json_to_jupytext(notebook_json: str) -> str:
    try:
        notebook = nbformat.reads(notebook_json, as_version=4)
        import jupytext

        return jupytext.writes(notebook, fmt="py:percent")
    except Exception as exc:
        raise RequirementsConfigError(f"Could not render notebook as Jupytext: {exc}") from exc


def parse_requirements(raw: Any, *, source: str) -> NotebookRequirements:
    if not isinstance(raw, dict):
        raise RequirementsConfigError(f"{source} must be a YAML/JSON object.")

    version = _optional_int(raw.get("version"), "version", default=1)
    if version != 1:
        raise RequirementsConfigError(f"{source} has unsupported version {version!r}.")

    runtime_raw = _optional_dict(raw.get("runtime"), "runtime")
    packages_raw = _optional_dict(raw.get("packages"), "packages")
    modal_raw = _optional_dict(raw.get("modal"), "modal")
    planner_raw = _optional_dict(raw.get("planner"), "planner")

    runtime = RuntimeRequirements(
        image=_optional_str(runtime_raw.get("image"), "runtime.image"),
        gpu=_optional_str(runtime_raw.get("gpu"), "runtime.gpu"),
        cpu=_optional_float(runtime_raw.get("cpu"), "runtime.cpu"),
        memory=_optional_int(runtime_raw.get("memory"), "runtime.memory"),
        timeout=_optional_int(
            runtime_raw.get("timeout"),
            "runtime.timeout",
            default=DEFAULT_TIMEOUT,
        )
        or DEFAULT_TIMEOUT,
        kernel_name=_optional_str(
            runtime_raw.get("kernel_name"),
            "runtime.kernel_name",
            default=DEFAULT_KERNEL_NAME,
        )
        or DEFAULT_KERNEL_NAME,
        python_version=_optional_str(
            runtime_raw.get("python_version"),
            "runtime.python_version",
            default=DEFAULT_PYTHON_VERSION,
        )
        or DEFAULT_PYTHON_VERSION,
        build_toolchain=_optional_bool(
            runtime_raw.get("build_toolchain"),
            "runtime.build_toolchain",
            default=True,
        ),
        pip_index_url=_optional_str(
            runtime_raw.get("pip_index_url"),
            "runtime.pip_index_url",
        ),
        pip_extra_index_urls=_string_list(
            runtime_raw.get("pip_extra_index_urls"),
            "runtime.pip_extra_index_urls",
        ),
    )
    packages = PackageRequirements(
        pip=_string_list(packages_raw.get("pip"), "packages.pip"),
        apt=_string_list(packages_raw.get("apt"), "packages.apt"),
    )
    if _pip_packages_need_build_toolchain(packages.pip):
        runtime = replace(runtime, build_toolchain=True)
    modal = ModalRequirements(
        secrets=_string_list(modal_raw.get("secrets"), "modal.secrets"),
        volumes=_string_list(modal_raw.get("volumes"), "modal.volumes"),
    )
    planner = PlannerMetadata(
        provider=_optional_str(
            planner_raw.get("provider"),
            "planner.provider",
            default="openrouter",
        )
        or "openrouter",
        model=_optional_str(
            planner_raw.get("model"),
            "planner.model",
            default=DEFAULT_OPENROUTER_MODEL,
        )
        or DEFAULT_OPENROUTER_MODEL,
        generated_at=_optional_str(planner_raw.get("generated_at"), "planner.generated_at"),
        confidence=_optional_float(planner_raw.get("confidence"), "planner.confidence"),
        notes=_string_list(planner_raw.get("notes"), "planner.notes"),
        source_hash=_optional_str(planner_raw.get("source_hash"), "planner.source_hash"),
    )

    return NotebookRequirements(
        version=version,
        runtime=runtime,
        packages=packages,
        modal=modal,
        planner=planner,
    )


def requirements_to_dict(requirements: NotebookRequirements) -> dict[str, Any]:
    return {
        "version": requirements.version,
        "runtime": {
            "image": requirements.runtime.image,
            "gpu": requirements.runtime.gpu,
            "cpu": requirements.runtime.cpu,
            "memory": requirements.runtime.memory,
            "timeout": requirements.runtime.timeout,
            "kernel_name": requirements.runtime.kernel_name,
            "python_version": requirements.runtime.python_version,
            "build_toolchain": requirements.runtime.build_toolchain,
            "pip_index_url": requirements.runtime.pip_index_url,
            "pip_extra_index_urls": list(requirements.runtime.pip_extra_index_urls),
        },
        "packages": {
            "pip": list(requirements.packages.pip),
            "apt": list(requirements.packages.apt),
        },
        "modal": {
            "secrets": list(requirements.modal.secrets),
            "volumes": list(requirements.modal.volumes),
        },
        "planner": {
            "provider": requirements.planner.provider,
            "model": requirements.planner.model,
            "generated_at": requirements.planner.generated_at,
            "confidence": requirements.planner.confidence,
            "notes": list(requirements.planner.notes),
            "source_hash": requirements.planner.source_hash,
        },
    }


def requirements_summary_lines(requirements: NotebookRequirements) -> list[str]:
    """Return a concise, stable human-readable summary of a requirements plan."""

    runtime = requirements.runtime
    packages = requirements.packages
    modal = requirements.modal
    planner = requirements.planner
    image = runtime.image or f"modal.Image.debian_slim(python_version='{runtime.python_version}')"
    gpu = runtime.gpu or "none"
    cpu = runtime.cpu if runtime.cpu is not None else "default"
    memory = f"{runtime.memory} MiB" if runtime.memory is not None else "default"
    confidence = (
        f"{planner.confidence:.2f}" if planner.confidence is not None else "unknown"
    )
    lines = [
        f"Planner: provider={planner.provider}, model={planner.model}, confidence={confidence}",
        (
            f"Runtime: image={image}, gpu={gpu}, cpu={cpu}, "
            f"memory={memory}, timeout={runtime.timeout}s"
        ),
        (
            f"Image setup: kernel={runtime.kernel_name}, "
            f"build_toolchain={runtime.build_toolchain}, pip={len(packages.pip)}, "
            f"apt={len(packages.apt)}"
        ),
        f"Modal attachments: secrets={len(modal.secrets)}, volumes={len(modal.volumes)}",
    ]
    lines.extend(f"Planner note: {note}" for note in planner.notes)
    return lines


def requirements_diff_lines(
    previous: NotebookRequirements,
    current: NotebookRequirements,
) -> list[str]:
    """Return changed top-level leaf fields between two requirements plans."""

    before = requirements_to_dict(previous)
    after = requirements_to_dict(current)
    paths: list[str] = []
    _collect_changed_paths(before, after, "", paths)
    return [f"Changed: {path}" for path in paths]


def _collect_changed_paths(
    before: Any,
    after: Any,
    prefix: str,
    paths: list[str],
) -> None:
    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before) | set(after))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            _collect_changed_paths(before.get(key), after.get(key), path, paths)
        return

    if before != after:
        paths.append(prefix)


def _plan_requirements_with_openrouter(
    notebook_text: str,
    model: str,
    api_key: str | None,
) -> dict[str, Any]:
    return _call_openrouter(notebook_text, model, api_key=api_key)


def _call_openrouter(
    notebook_text: str,
    model: str,
    *,
    api_key: str | None = None,
    retries: int = DEFAULT_OPENROUTER_RETRIES,
    timeout_seconds: int = DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    selected_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not selected_api_key:
        raise RequirementsConfigError(
            "No companion requirements file was found and OPENROUTER_API_KEY is not set."
        )

    payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": DEFAULT_OPENROUTER_MAX_COMPLETION_TOKENS,
        "reasoning": {
            "effort": DEFAULT_OPENROUTER_REASONING_EFFORT,
            "exclude": True,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "runbook_execution_requirements",
                "strict": True,
                "schema": _openrouter_response_schema(),
            },
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You infer minimal Modal remote execution requirements for a "
                    "Jupyter notebook. Return JSON only. Do not invent secrets, "
                    "volumes, private images, or credentials. Prefer a public base "
                    "image and explicit pip/apt packages that are sufficient for the "
                    "notebook to run. Use null when a resource is not needed. Set "
                    "runtime.build_toolchain to true when packages may compile native "
                    "or CUDA code at install or runtime, including Triton, Unsloth, "
                    "xFormers, bitsandbytes, flash-attn, or DeepSpeed. Set "
                    "planner.source_hash to null; Runbook fills it locally."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Infer the companion requirements file for this notebook. The "
                    "notebook is provided in Jupytext percent format.\n\n"
                    f"{notebook_text}"
                ),
            },
        ],
    }
    response_body = _openrouter_request_with_retries(
        payload,
        selected_api_key,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )

    try:
        completion = json.loads(response_body)
        return _parse_openrouter_completion(completion)
    except Exception as exc:
        raise RequirementsConfigError(
            f"OpenRouter returned an invalid requirements response: {exc}"
        ) from exc


def _openrouter_request_with_retries(
    payload: dict[str, Any],
    api_key: str,
    *,
    retries: int,
    timeout_seconds: int,
) -> str:
    last_error: BaseException | None = None
    attempts = max(retries, 0) + 1
    for attempt in range(attempts):
        request = urllib.request.Request(
            os.environ.get("RUNBOOK_OPENROUTER_URL", OPENROUTER_CHAT_COMPLETIONS_URL),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Runbook",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if not _should_retry_http_status(exc.code) or attempt == attempts - 1:
                raise RequirementsConfigError(
                    f"OpenRouter request failed: {exc.code} {body}"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            if attempt == attempts - 1:
                raise RequirementsConfigError(f"OpenRouter request failed: {exc}") from exc
            last_error = exc
        time.sleep(min(0.25 * (2**attempt), 2.0))
    raise RequirementsConfigError(f"OpenRouter request failed: {last_error}")


def _should_retry_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _parse_openrouter_completion(completion: dict[str, Any]) -> dict[str, Any]:
    choice = completion["choices"][0]
    message = choice["message"]
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(_content_part_text(item) for item in content)
    if not isinstance(content, str) or not content.strip():
        finish_reason = choice.get("finish_reason")
        raise TypeError(
            "message content is empty or not text"
            f" (finish_reason={finish_reason!r})"
        )
    return json.loads(_strip_json_fences(content))


def _content_part_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    text = item.get("text")
    if isinstance(text, str):
        return text
    content = item.get("content")
    if isinstance(content, str):
        return content
    return ""


def _openrouter_response_schema() -> dict[str, Any]:
    string_or_null = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    number_or_null = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    integer_or_null = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    boolean = {"type": "boolean"}
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "runtime", "packages", "modal", "planner"],
        "properties": {
            "version": {"type": "integer", "const": 1},
            "runtime": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "image",
                    "gpu",
                    "cpu",
                    "memory",
                    "timeout",
                    "kernel_name",
                    "python_version",
                    "build_toolchain",
                    "pip_index_url",
                    "pip_extra_index_urls",
                ],
                "properties": {
                    "image": string_or_null,
                    "gpu": string_or_null,
                    "cpu": number_or_null,
                    "memory": integer_or_null,
                    "timeout": {"type": "integer"},
                    "kernel_name": {"type": "string"},
                    "python_version": {"type": "string"},
                    "build_toolchain": boolean,
                    "pip_index_url": string_or_null,
                    "pip_extra_index_urls": string_array,
                },
            },
            "packages": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pip", "apt"],
                "properties": {"pip": string_array, "apt": string_array},
            },
            "modal": {
                "type": "object",
                "additionalProperties": False,
                "required": ["secrets", "volumes"],
                "properties": {"secrets": string_array, "volumes": string_array},
            },
            "planner": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "provider",
                    "model",
                    "generated_at",
                    "confidence",
                    "notes",
                    "source_hash",
                ],
                "properties": {
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                    "generated_at": string_or_null,
                    "confidence": number_or_null,
                    "notes": string_array,
                    "source_hash": string_or_null,
                },
            },
        },
    }


def _with_planner_defaults(
    requirements: NotebookRequirements,
    model: str,
    *,
    source_hash: str,
) -> NotebookRequirements:
    planner = PlannerMetadata(
        provider=requirements.planner.provider or "openrouter",
        model=model,
        generated_at=requirements.planner.generated_at
        or datetime.now(timezone.utc).isoformat(),
        confidence=requirements.planner.confidence,
        notes=requirements.planner.notes,
        source_hash=source_hash,
    )
    return NotebookRequirements(
        version=requirements.version,
        runtime=requirements.runtime,
        packages=requirements.packages,
        modal=requirements.modal,
        planner=planner,
    )


def notebook_source_hash(notebook_json: str) -> str:
    return hashlib.sha256(notebook_json.encode("utf-8")).hexdigest()


def _ensure_requirements_current(
    requirements: NotebookRequirements,
    notebook_json: str,
    path: Path,
) -> None:
    source_hash = requirements.planner.source_hash
    if source_hash is None:
        return
    current_hash = notebook_source_hash(notebook_json)
    if source_hash != current_hash:
        raise RequirementsConfigError(
            f"{path} was generated for a different notebook revision. "
            "Delete it or rerun with --regenerate-requirements."
        )


def _prepare_notebook_text_for_planner(notebook_text: str) -> str:
    limit = int(
        os.environ.get(
            "RUNBOOK_OPENROUTER_MAX_NOTEBOOK_CHARS",
            DEFAULT_OPENROUTER_MAX_NOTEBOOK_CHARS,
        )
    )
    if len(notebook_text) > limit:
        raise RequirementsConfigError(
            "Notebook is too large to send to OpenRouter for automatic planning "
            f"({len(notebook_text)} characters; limit {limit}). Create a companion "
            "requirements YAML manually or raise RUNBOOK_OPENROUTER_MAX_NOTEBOOK_CHARS."
        )
    return _redact_planner_text(notebook_text)


def _redact_planner_text(notebook_text: str) -> str:
    redacted_lines: list[str] = []
    sensitive_markers = ("api_key", "apikey", "token", "secret", "password")
    for line in notebook_text.splitlines():
        normalized = line.lower()
        if "=" in line and any(marker in normalized for marker in sensitive_markers):
            key = line.split("=", 1)[0].rstrip()
            redacted_lines.append(f"{key}= '<redacted>'")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


def _optional_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RequirementsConfigError(f"{field_name} must be an object.")
    return value


def _optional_str(value: Any, field_name: str, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        raise RequirementsConfigError(f"{field_name} must be a string or null.")
    return value


def _optional_int(value: Any, field_name: str, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequirementsConfigError(f"{field_name} must be an integer.")
    return value


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequirementsConfigError(f"{field_name} must be a number or null.")
    return float(value)


def _optional_bool(value: Any, field_name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RequirementsConfigError(f"{field_name} must be a boolean.")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RequirementsConfigError(f"{field_name} must be a list of strings.")
    return _dedupe(value)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _pip_packages_need_build_toolchain(packages: list[str]) -> bool:
    return any(
        _normalized_pip_package_name(package) in BUILD_TOOLCHAIN_PIP_PACKAGES
        for package in packages
    )


def _normalized_pip_package_name(package: str) -> str:
    name = package.strip()
    if "://" in name:
        return ""
    name = re.split(r"\s*(?:\[|==|!=|~=|>=|<=|>|<|;)", name, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _strip_json_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
