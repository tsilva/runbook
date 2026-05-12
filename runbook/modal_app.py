from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

Event = dict[str, Any]


@dataclass(frozen=True)
class ModalRunOptions:
    timeout: int = 3600
    gpu: str | None = None
    cpu: float | None = None
    memory: int | None = None
    secrets: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    image: str | None = None
    allow_errors: bool = False
    kernel_name: str = "python3"


class ModalSetupError(RuntimeError):
    """Raised when a Modal runner cannot be configured."""


def _runbook_remote_runner(
    uploaded_notebook_json: str,
    allow_errors: bool,
    cell_timeout: int,
    kernel_name: str,
    debug: dict[str, Any],
):
    from runbook.execute import execute_notebook_events

    yield from execute_notebook_events(
        uploaded_notebook_json,
        allow_errors=allow_errors,
        timeout=cell_timeout,
        kernel_name=kernel_name,
        debug=debug,
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
    image = _build_image(modal, options.image)
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
            "image": options.image or "modal.Image.debian_slim(python_version='3.11')",
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
                debug,
            )
    except Exception as exc:  # pragma: no cover - Modal integration behavior
        raise ModalSetupError(f"Modal execution failed: {exc}") from exc


def _build_image(modal: Any, image_name: str | None) -> Any:
    if image_name:
        image = modal.Image.from_registry(image_name)
    else:
        image = modal.Image.debian_slim(python_version="3.11")
    return image.pip_install("nbformat", "nbclient", "ipykernel").add_local_python_source(
        "runbook"
    )


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
