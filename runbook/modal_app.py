from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from runbook.events import Event


@dataclass(frozen=True)
class ModalRunOptions:
    timeout: int = 3600
    gpu: str | None = None
    cpu: float | None = None
    memory: int | None = None
    secrets: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    image: str | None = None
    pip_packages: list[str] = field(default_factory=list)
    apt_packages: list[str] = field(default_factory=list)
    allow_errors: bool = False
    kernel_name: str = "python3"
    python_version: str = "3.11"
    build_toolchain: bool = True
    pip_index_url: str | None = None
    pip_extra_index_urls: list[str] = field(default_factory=list)
    workdir: str = "/tmp/runbook"


class ModalSetupError(RuntimeError):
    """Raised when a Modal runner cannot be configured."""


@dataclass(frozen=True)
class ModalPreflightReport:
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def preflight_modal_run(options: ModalRunOptions) -> ModalPreflightReport:
    """Validate local Modal configuration and the static execution plan."""

    checks: list[str] = []
    warnings: list[str] = []
    try:
        import modal
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModalSetupError(
            "Modal is not installed. Install runbook with its dependencies and run "
            "`python3 -m modal setup`."
        ) from exc

    _build_image(
        modal,
        options.image,
        pip_packages=options.pip_packages,
        apt_packages=options.apt_packages,
        python_version=options.python_version,
        build_toolchain=options.build_toolchain,
        pip_index_url=options.pip_index_url,
        pip_extra_index_urls=options.pip_extra_index_urls,
    )
    checks.append("Modal image definition can be constructed locally.")

    if options.secrets:
        for name in options.secrets:
            if not name.strip():
                raise ModalSetupError("Secret names cannot be empty.")
            modal.Secret.from_name(name)
        checks.append(f"Prepared {len(options.secrets)} Modal Secret reference(s).")
    else:
        checks.append("No Modal Secret references requested.")

    if options.volumes:
        _parse_volumes(modal, options.volumes)
        checks.append(f"Prepared {len(options.volumes)} Modal Volume mount(s).")
    else:
        checks.append("No Modal Volume mounts requested.")

    if options.gpu:
        warning = _gpu_name_warning(options.gpu)
        if warning:
            warnings.append(warning)
        else:
            checks.append(f"GPU request {options.gpu!r} has a recognized shape.")
    else:
        checks.append("CPU execution selected.")

    checks.append(
        "Modal authentication and remote resource existence will be verified by Modal "
        "when execution starts."
    )
    return ModalPreflightReport(checks=checks, warnings=warnings)


def _runbook_remote_runner(
    uploaded_notebook_json: str,
    allow_errors: bool,
    cell_timeout: int,
    kernel_name: str,
    workdir: str,
    debug: dict[str, Any],
):
    from runbook.execute import execute_notebook_events

    yield from execute_notebook_events(
        uploaded_notebook_json,
        allow_errors=allow_errors,
        timeout=cell_timeout,
        kernel_name=kernel_name,
        debug=debug,
        workdir=workdir,
    )


def stream_remote_events(notebook_json: str, options: ModalRunOptions) -> Iterator[Event]:
    """Build an ephemeral Modal app and stream events from remote_gen."""

    try:
        import modal
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModalSetupError(
            "Modal is not installed. Install runbook with its dependencies and run "
            "`python3 -m modal setup`."
        ) from exc

    app_name = "runbook"
    function_name = "runbook_remote_runner"
    app = modal.App(app_name)
    image = _build_image(
        modal,
        options.image,
        pip_packages=options.pip_packages,
        apt_packages=options.apt_packages,
        python_version=options.python_version,
        build_toolchain=options.build_toolchain,
        pip_index_url=options.pip_index_url,
        pip_extra_index_urls=options.pip_extra_index_urls,
    )
    function_kwargs: dict[str, Any] = {
        "image": image,
        "timeout": options.timeout,
        "name": function_name,
    }
    if options.gpu:
        function_kwargs["gpu"] = options.gpu
    if options.cpu is not None:
        function_kwargs["cpu"] = options.cpu
    if options.memory is not None:
        function_kwargs["memory"] = options.memory
    if options.secrets:
        function_kwargs["secrets"] = [modal.Secret.from_name(name) for name in options.secrets]
    if options.volumes:
        function_kwargs["volumes"] = _parse_volumes(modal, options.volumes)

    runbook_remote_runner = app.function(**function_kwargs)(_runbook_remote_runner)

    debug = {
        "app_name": app_name,
        "function_name": function_name,
        "dashboard_url": "https://modal.com/apps",
        "resources": {
            "gpu": options.gpu,
            "cpu": options.cpu,
            "memory": options.memory,
            "timeout": options.timeout,
            "image": options.image
            or f"modal.Image.debian_slim(python_version='{options.python_version}')",
            "pip_packages": options.pip_packages,
            "apt_packages": options.apt_packages,
            "python_version": options.python_version,
            "build_toolchain": options.build_toolchain,
            "pip_index_url": options.pip_index_url,
            "pip_extra_index_urls": options.pip_extra_index_urls,
            "workdir": options.workdir,
            "secrets": options.secrets,
            "volumes": options.volumes,
        },
    }

    try:
        with app.run():
            yield from runbook_remote_runner.remote_gen(
                notebook_json,
                options.allow_errors,
                options.timeout,
                options.kernel_name,
                options.workdir,
                debug,
            )
    except Exception as exc:  # pragma: no cover - Modal integration behavior
        raise ModalSetupError(f"Modal execution failed: {exc}") from exc


def _build_image(
    modal: Any,
    image_name: str | None,
    *,
    pip_packages: list[str] | None = None,
    apt_packages: list[str] | None = None,
    python_version: str = "3.11",
    build_toolchain: bool = True,
    pip_index_url: str | None = None,
    pip_extra_index_urls: list[str] | None = None,
) -> Any:
    if image_name:
        image = modal.Image.from_registry(image_name)
    else:
        image = modal.Image.debian_slim(python_version=python_version)
    base_apt = ["build-essential"] if build_toolchain else []
    apt = _dedupe([*base_apt, *(apt_packages or [])])
    pip = _dedupe(["nbformat", "nbclient", "ipykernel", *(pip_packages or [])])
    if apt:
        image = image.apt_install(*apt)
    pip_kwargs: dict[str, Any] = {}
    if pip_index_url:
        pip_kwargs["index_url"] = pip_index_url
    if pip_extra_index_urls:
        pip_kwargs["extra_index_url"] = pip_extra_index_urls
    return image.pip_install(*pip, **pip_kwargs).add_local_python_source("runbook")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _parse_volumes(modal: Any, volume_specs: list[str]) -> dict[str, Any]:
    volumes: dict[str, Any] = {}
    for spec in volume_specs:
        if ":" not in spec:
            raise ModalSetupError(
                f"Invalid volume spec {spec!r}. Expected NAME:/mount/path."
            )
        name, mount_path = spec.split(":", 1)
        if not name or not mount_path.startswith("/"):
            raise ModalSetupError(
                f"Invalid volume spec {spec!r}. Expected NAME:/mount/path."
            )
        volumes[mount_path] = modal.Volume.from_name(name)
    return volumes


def _gpu_name_warning(gpu: str) -> str | None:
    normalized = gpu.upper()
    recognized_prefixes = (
        "T4",
        "L4",
        "A10",
        "A100",
        "H100",
        "H200",
        "L40",
        "L40S",
        "B200",
    )
    if normalized.startswith(recognized_prefixes):
        return None
    return (
        f"GPU request {gpu!r} is not one of Runbook's recognized Modal GPU shapes; "
        "Modal may still accept it."
    )
