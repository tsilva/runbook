from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import yaml


DEFAULT_TIMEOUT = 3600
DEFAULT_KERNEL_NAME = "python3"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.5"
DEFAULT_OPENROUTER_REASONING_EFFORT = "high"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class RuntimeRequirements:
    image: str | None = None
    gpu: str | None = None
    cpu: float | None = None
    memory: int | None = None
    timeout: int = DEFAULT_TIMEOUT
    kernel_name: str = DEFAULT_KERNEL_NAME


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
) -> RequirementsLoadResult:
    path = companion_requirements_path(input_path)
    if path.exists():
        return RequirementsLoadResult(
            path=path,
            requirements=load_requirements(path),
            generated=False,
        )

    requirements = generate_requirements(notebook_json, model=model, api_key=api_key)
    write_requirements(path, requirements)
    return RequirementsLoadResult(path=path, requirements=requirements, generated=True)


def load_requirements(path: Path) -> NotebookRequirements:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RequirementsConfigError(f"Could not read requirements file {path}: {exc}") from exc
    return parse_requirements(raw, source=str(path))


def write_requirements(path: Path, requirements: NotebookRequirements) -> None:
    path.write_text(
        yaml.safe_dump(requirements_to_dict(requirements), sort_keys=False),
        encoding="utf-8",
    )


def generate_requirements(
    notebook_json: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> NotebookRequirements:
    selected_model = model or os.environ.get("RUNBOOK_OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    notebook_text = notebook_json_to_jupytext(notebook_json)
    raw_plan = _call_openrouter(notebook_text, selected_model, api_key=api_key)
    requirements = parse_requirements(raw_plan, source="OpenRouter response")
    return _with_planner_defaults(requirements, selected_model)


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
        ),
        kernel_name=_optional_str(
            runtime_raw.get("kernel_name"),
            "runtime.kernel_name",
            default=DEFAULT_KERNEL_NAME,
        ),
    )
    packages = PackageRequirements(
        pip=_string_list(packages_raw.get("pip"), "packages.pip"),
        apt=_string_list(packages_raw.get("apt"), "packages.apt"),
    )
    modal = ModalRequirements(
        secrets=_string_list(modal_raw.get("secrets"), "modal.secrets"),
        volumes=_string_list(modal_raw.get("volumes"), "modal.volumes"),
    )
    planner = PlannerMetadata(
        provider=_optional_str(
            planner_raw.get("provider"),
            "planner.provider",
            default="openrouter",
        ),
        model=_optional_str(
            planner_raw.get("model"),
            "planner.model",
            default=DEFAULT_OPENROUTER_MODEL,
        ),
        generated_at=_optional_str(planner_raw.get("generated_at"), "planner.generated_at"),
        confidence=_optional_float(planner_raw.get("confidence"), "planner.confidence"),
        notes=_string_list(planner_raw.get("notes"), "planner.notes"),
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
        },
    }


def _call_openrouter(
    notebook_text: str,
    model: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    selected_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not selected_api_key:
        raise RequirementsConfigError(
            "No companion requirements file was found and OPENROUTER_API_KEY is not set."
        )

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2400,
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
                    "notebook to run. Use null when a resource is not needed."
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
    request = urllib.request.Request(
        os.environ.get("RUNBOOK_OPENROUTER_URL", OPENROUTER_CHAT_COMPLETIONS_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {selected_api_key}",
            "Content-Type": "application/json",
            "X-Title": "Runbook",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RequirementsConfigError(f"OpenRouter request failed: {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RequirementsConfigError(f"OpenRouter request failed: {exc}") from exc

    try:
        completion = json.loads(response_body)
        content = completion["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str):
            raise TypeError("message content is not a string")
        return json.loads(_strip_json_fences(content))
    except Exception as exc:
        raise RequirementsConfigError(
            f"OpenRouter returned an invalid requirements response: {exc}"
        ) from exc


def _openrouter_response_schema() -> dict[str, Any]:
    string_or_null = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    number_or_null = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    integer_or_null = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
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
                "required": ["image", "gpu", "cpu", "memory", "timeout", "kernel_name"],
                "properties": {
                    "image": string_or_null,
                    "gpu": string_or_null,
                    "cpu": number_or_null,
                    "memory": integer_or_null,
                    "timeout": {"type": "integer"},
                    "kernel_name": {"type": "string"},
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
                "required": ["provider", "model", "generated_at", "confidence", "notes"],
                "properties": {
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                    "generated_at": string_or_null,
                    "confidence": number_or_null,
                    "notes": string_array,
                },
            },
        },
    }


def _with_planner_defaults(
    requirements: NotebookRequirements,
    model: str,
) -> NotebookRequirements:
    planner = PlannerMetadata(
        provider=requirements.planner.provider or "openrouter",
        model=model,
        generated_at=requirements.planner.generated_at
        or datetime.now(timezone.utc).isoformat(),
        confidence=requirements.planner.confidence,
        notes=requirements.planner.notes,
    )
    return NotebookRequirements(
        version=requirements.version,
        runtime=requirements.runtime,
        packages=requirements.packages,
        modal=requirements.modal,
        planner=planner,
    )


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
