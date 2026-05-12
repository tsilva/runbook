from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import nbformat
import typer
from rich.console import Console

from runbook.convert import NotebookConversionError, default_output_path, read_notebook
from runbook.files import atomic_write_text
from runbook.modal_app import (
    ModalRunOptions,
    ModalSetupError,
    preflight_modal_run,
    stream_remote_events,
)
from runbook.progress import (
    NotebookProgress,
    print_debug_info,
    print_failure_summary,
    print_startup_failure_summary,
    print_success_summary,
)
from runbook.requirements_plan import (
    DEFAULT_KERNEL_NAME,
    DEFAULT_PYTHON_VERSION,
    DEFAULT_TIMEOUT,
    ModalRequirements,
    NotebookRequirements,
    PackageRequirements,
    RequirementsConfigError,
    RequirementsLoadResult,
    RuntimeRequirements,
    companion_requirements_path,
    load_or_generate_requirements,
    requirements_diff_lines,
    requirements_summary_lines,
)
from runbook.settings import (
    OpenRouterSettings,
    init_settings_dir,
    load_openrouter_settings,
    runbook_env_path,
    save_openrouter_settings,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(help="Input .ipynb or Jupytext .py file.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output executed .ipynb path."),
    ] = None,
    gpu: Annotated[
        str | None,
        typer.Option("--gpu", help="Modal GPU type, for example A10. Omit for CPU."),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Modal function and per-cell timeout in seconds."),
    ] = None,
    cpu: Annotated[
        float | None,
        typer.Option("--cpu", help="Modal CPU core request."),
    ] = None,
    memory: Annotated[
        int | None,
        typer.Option("--memory", help="Modal memory request in MiB."),
    ] = None,
    secret: Annotated[
        list[str] | None,
        typer.Option("--secret", help="Modal Secret name. Repeatable."),
    ] = None,
    volume: Annotated[
        list[str] | None,
        typer.Option("--volume", help="Modal Volume mount NAME:/mount/path. Repeatable."),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option("--image", help="Public registry image to use as the Modal base image."),
    ] = None,
    pip_package: Annotated[
        list[str] | None,
        typer.Option("--pip-package", help="Additional pip package. Repeatable."),
    ] = None,
    apt_package: Annotated[
        list[str] | None,
        typer.Option("--apt-package", help="Additional apt package. Repeatable."),
    ] = None,
    allow_errors: Annotated[
        bool,
        typer.Option("--allow-errors", help="Continue executing after cell errors."),
    ] = False,
    kernel_name: Annotated[
        str | None,
        typer.Option("--kernel-name", help="Jupyter kernel name."),
    ] = None,
    regenerate_requirements: Annotated[
        bool,
        typer.Option(
            "--regenerate-requirements",
            help="Regenerate the companion requirements file even if one exists.",
        ),
    ] = False,
    python_version: Annotated[
        str | None,
        typer.Option("--python-version", help="Python version for the default Modal image."),
    ] = None,
    build_toolchain: Annotated[
        bool | None,
        typer.Option(
            "--build-toolchain/--no-build-toolchain",
            help="Install build-essential in the Modal image.",
        ),
    ] = None,
    pip_index_url: Annotated[
        str | None,
        typer.Option("--pip-index-url", help="Custom pip index URL for Modal image installs."),
    ] = None,
    pip_extra_index_url: Annotated[
        list[str] | None,
        typer.Option(
            "--pip-extra-index-url",
            help="Additional pip index URL for Modal image installs. Repeatable.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Resolve requirements and validate the Modal plan without executing the notebook.",
        ),
    ] = False,
) -> None:
    """Execute a notebook remotely on Modal and write an executed .ipynb."""

    console = Console(stderr=True)
    log = TerminalLog(console)
    log.info("Runbook run started.")

    settings_started = perf_counter()
    init_settings_dir()
    log.done("Settings directory initialized", settings_started)

    conversion_started = perf_counter()
    log.info(f"Reading notebook from {input_path}.")
    try:
        converted = read_notebook(input_path)
    except NotebookConversionError as exc:
        console.print(f"Notebook conversion failed: {exc}")
        raise typer.Exit(1) from exc
    log.done(f"Notebook ready from {converted.source_path}", conversion_started)

    requirements_started = perf_counter()
    try:
        requirements_result = _load_or_create_requirements(
            console,
            log,
            converted.source_path,
            converted.notebook_json,
            timeout=timeout,
            gpu=gpu,
            cpu=cpu,
            memory=memory,
            secrets=secret,
            volumes=volume,
            image=image,
            pip_packages=pip_package,
            apt_packages=apt_package,
            kernel_name=kernel_name,
            regenerate_requirements=regenerate_requirements,
            python_version=python_version,
            build_toolchain=build_toolchain,
            pip_index_url=pip_index_url,
            pip_extra_index_urls=pip_extra_index_url,
        )
    except RequirementsConfigError as exc:
        console.print(f"Execution requirements failed: {exc}")
        raise typer.Exit(1) from exc
    log.done("Execution requirements resolved", requirements_started)

    if requirements_result.generated:
        log.info(f"Wrote execution requirements to {requirements_result.path}.")
        _print_requirements_plan(log, requirements_result)
    elif requirements_result.path.exists():
        log.info(f"Using execution requirements from {requirements_result.path}.")
    else:
        log.info("Using CLI-provided execution requirements.")

    output_path = (output or default_output_path(converted.source_path)).expanduser().resolve()
    options_started = perf_counter()
    options = _merge_options(
        requirements_result.requirements,
        timeout=timeout,
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        secrets=secret,
        volumes=volume,
        image=image,
        pip_packages=pip_package,
        apt_packages=apt_package,
        allow_errors=allow_errors,
        kernel_name=kernel_name,
        python_version=python_version,
        build_toolchain=build_toolchain,
        pip_index_url=pip_index_url,
        pip_extra_index_urls=pip_extra_index_url,
    )
    log.done("Modal run options resolved", options_started)

    preflight_started = perf_counter()
    try:
        _run_preflight(log, output_path, options)
    except ModalSetupError as exc:
        log.done("Preflight failed", preflight_started)
        console.print(f"Preflight failed: {exc}")
        raise typer.Exit(1) from exc
    log.done("Preflight completed", preflight_started)

    if dry_run:
        console.print("Dry run complete; remote execution skipped.")
        return

    _print_remote_execution_start(log, options)

    notebook_data: str | None = None
    debug: dict | None = None
    failure: dict | None = None
    startup_failure: dict | None = None
    completed = 0
    total_cells = 0
    allowed_error_count = 0
    finished = False
    modal_setup_started = perf_counter()
    remote_execution_started: float | None = None
    cell_started_at: dict[int, float] = {}
    modal_setup_reported = False

    try:
        with tempfile.TemporaryDirectory(prefix="runbook-"):
            with NotebookProgress(console) as progress:
                for event in stream_remote_events(converted.notebook_json, options):
                    event_data: dict[str, Any] = dict(event)
                    kind = event_data.get("event")
                    if not modal_setup_reported:
                        log.done("Modal setup and image preparation", modal_setup_started)
                        modal_setup_reported = True
                    if kind == "started":
                        remote_execution_started = perf_counter()
                        total_cells = int(event_data.get("total_cells", 0))
                        event_debug = event_data.get("debug")
                        debug = event_debug if isinstance(event_debug, dict) else debug
                        print_debug_info(console, "Modal run started", debug)
                        log.info(
                            "Remote notebook execution started with "
                            f"{total_cells} executable cell(s)."
                        )
                        progress.start(total_cells)
                    elif kind == "cell_started":
                        cell = int(event_data.get("cell", 0))
                        event_total = int(event_data.get("total_cells", total_cells))
                        cell_started_at[cell] = perf_counter()
                        log.info(f"Cell {cell}/{event_total} started.")
                        progress.current(cell, event_total)
                    elif kind == "cell_finished":
                        completed = int(event_data.get("completed", completed))
                        total_cells = int(event_data.get("total_cells", total_cells))
                        cell = int(event_data.get("cell", completed))
                        started_at = cell_started_at.pop(cell, None)
                        elapsed = (
                            f" in {_format_duration(perf_counter() - started_at)}"
                            if started_at is not None
                            else ""
                        )
                        log.info(f"Cell {cell}/{total_cells} finished{elapsed}.")
                        progress.update(completed, total_cells)
                    elif kind == "cell_failed":
                        cell = int(event_data.get("cell", 0))
                        event_total = int(event_data.get("total_cells", total_cells))
                        started_at = cell_started_at.pop(cell, None)
                        elapsed = (
                            f" after {_format_duration(perf_counter() - started_at)}"
                            if started_at is not None
                            else ""
                        )
                        log.info(
                            f"Cell {cell}/{event_total} failed{elapsed}: "
                            f"{event_data.get('error_type')}: {event_data.get('message')}"
                        )
                        if event_data.get("allowed"):
                            allowed_error_count += 1
                        else:
                            failure = event_data
                    elif kind == "cell_output":
                        _print_cell_output(log, event_data)
                    elif kind == "startup_failed":
                        startup_failure = event_data
                        log.info(
                            "Notebook startup failed before cell execution: "
                            f"{event_data.get('error_type')}: {event_data.get('message')}"
                        )
                    elif kind == "finished":
                        finished = True
                        completed = int(event_data.get("completed", completed))
                        total_cells = int(event_data.get("total_cells", total_cells))
                        if remote_execution_started is not None:
                            log.done("Remote notebook execution", remote_execution_started)
                        progress.update(completed, total_cells)
                    elif kind == "notebook":
                        if event_data.get("format") == "ipynb-json":
                            status = _notebook_status(
                                finished=finished,
                                failure=failure,
                                startup_failure=startup_failure,
                            )
                            notebook_data = _attach_run_manifest(
                                str(event_data.get("data", "")),
                                requirements_result,
                                options,
                                debug,
                                status=status,
                                completed=completed,
                                total_cells=total_cells,
                            )
                            log.info(
                                "Received output notebook payload "
                                f"({_format_bytes(len(notebook_data.encode('utf-8')))})."
                            )
    except ModalSetupError as exc:
        if not modal_setup_reported:
            log.done("Modal setup failed", modal_setup_started)
        _write_notebook_if_available(output_path, notebook_data, log, partial=True)
        console.print(f"Modal setup/execution failed: {exc}")
        if debug:
            print_debug_info(console, "Modal run", debug)
        raise typer.Exit(1) from exc
    except Exception as exc:
        if not modal_setup_reported:
            log.done("Remote execution failed before first Modal event", modal_setup_started)
        _write_notebook_if_available(output_path, notebook_data, log, partial=True)
        console.print(f"Remote execution failed: {exc}")
        if debug:
            print_debug_info(console, "Modal run", debug)
        raise typer.Exit(1) from exc

    if not modal_setup_reported:
        log.done("Modal setup and image preparation ended without events", modal_setup_started)

    wrote_output = _write_notebook_if_available(output_path, notebook_data, log)
    written_path = output_path if wrote_output else None

    if startup_failure is not None:
        print_startup_failure_summary(console, written_path, startup_failure, debug)
        raise typer.Exit(1)

    if failure is not None:
        print_failure_summary(console, written_path, failure, debug)
        raise typer.Exit(1)

    if not finished:
        console.print("Remote execution ended before a finished event was received.")
        if written_path is not None:
            console.print(f"Wrote partial notebook to {written_path}.")
        if debug:
            print_debug_info(console, "Modal run", debug)
        raise typer.Exit(1)

    if written_path is None:
        console.print("Remote execution finished without returning an output notebook.")
        raise typer.Exit(1)

    print_success_summary(
        console,
        written_path,
        completed,
        total_cells,
        debug,
        allowed_error_count=allowed_error_count,
    )


def _merge_options(
    requirements: NotebookRequirements,
    *,
    timeout: int | None,
    gpu: str | None,
    cpu: float | None,
    memory: int | None,
    secrets: list[str] | None,
    volumes: list[str] | None,
    image: str | None,
    pip_packages: list[str] | None,
    apt_packages: list[str] | None,
    allow_errors: bool,
    kernel_name: str | None,
    python_version: str | None,
    build_toolchain: bool | None,
    pip_index_url: str | None,
    pip_extra_index_urls: list[str] | None,
) -> ModalRunOptions:
    runtime = requirements.runtime
    modal = requirements.modal
    packages = requirements.packages
    return ModalRunOptions(
        timeout=timeout if timeout is not None else runtime.timeout or DEFAULT_TIMEOUT,
        gpu=gpu if gpu is not None else runtime.gpu,
        cpu=cpu if cpu is not None else runtime.cpu,
        memory=memory if memory is not None else runtime.memory,
        secrets=secrets if secrets is not None else modal.secrets,
        volumes=volumes if volumes is not None else modal.volumes,
        image=image if image is not None else runtime.image,
        pip_packages=_dedupe([*packages.pip, *(pip_packages or [])]),
        apt_packages=_dedupe([*packages.apt, *(apt_packages or [])]),
        allow_errors=allow_errors,
        kernel_name=kernel_name
        if kernel_name is not None
        else runtime.kernel_name or DEFAULT_KERNEL_NAME,
        python_version=python_version
        if python_version is not None
        else runtime.python_version or DEFAULT_PYTHON_VERSION,
        build_toolchain=build_toolchain
        if build_toolchain is not None
        else runtime.build_toolchain,
        pip_index_url=pip_index_url if pip_index_url is not None else runtime.pip_index_url,
        pip_extra_index_urls=_dedupe(
            [*runtime.pip_extra_index_urls, *(pip_extra_index_urls or [])]
        ),
        workdir=f"/tmp/runbook-{uuid.uuid4().hex}",
    )


def _load_or_create_requirements(
    console: Console,
    log: TerminalLog,
    input_path: Path,
    notebook_json: str,
    *,
    timeout: int | None,
    gpu: str | None,
    cpu: float | None,
    memory: int | None,
    secrets: list[str] | None,
    volumes: list[str] | None,
    image: str | None,
    pip_packages: list[str] | None,
    apt_packages: list[str] | None,
    kernel_name: str | None,
    regenerate_requirements: bool,
    python_version: str | None,
    build_toolchain: bool | None,
    pip_index_url: str | None,
    pip_extra_index_urls: list[str] | None,
) -> RequirementsLoadResult:
    requirements_path = companion_requirements_path(input_path)
    if requirements_path.exists() and not regenerate_requirements:
        started = perf_counter()
        log.info(f"Loading execution requirements from {requirements_path}.")
        result = load_or_generate_requirements(input_path, notebook_json)
        log.done("Execution requirements loaded", started)
        return result

    settings_started = perf_counter()
    settings = _load_or_prompt_openrouter_settings(console)
    log.done("OpenRouter settings resolved", settings_started)
    if settings.api_key:
        started = perf_counter()
        log.info(
            f"Generating execution requirements with OpenRouter ({settings.model})."
        )
        result = load_or_generate_requirements(
            input_path,
            notebook_json,
            api_key=settings.api_key,
            model=settings.model,
            regenerate=regenerate_requirements,
        )
        log.done("Execution requirements generated", started)
        return result

    if not image:
        raise RequirementsConfigError(
            "No companion requirements file exists and OpenRouter settings were "
            "not provided. Pass --image to run without generating requirements, "
            "plus --gpu/--pip-package/--apt-package as needed."
        )

    log.info("No config generation needed; using CLI-provided execution requirements.")
    return RequirementsLoadResult(
        path=requirements_path,
        requirements=NotebookRequirements(
            runtime=RuntimeRequirements(
                image=image,
                gpu=gpu,
                cpu=cpu,
                memory=memory,
                timeout=timeout or DEFAULT_TIMEOUT,
                kernel_name=kernel_name or DEFAULT_KERNEL_NAME,
                python_version=python_version or DEFAULT_PYTHON_VERSION,
                build_toolchain=True if build_toolchain is None else build_toolchain,
                pip_index_url=pip_index_url,
                pip_extra_index_urls=pip_extra_index_urls or [],
            ),
            packages=PackageRequirements(
                pip=pip_packages or [],
                apt=apt_packages or [],
            ),
            modal=ModalRequirements(
                secrets=secrets or [],
                volumes=volumes or [],
            ),
        ),
        generated=False,
    )


def _print_remote_execution_start(log: TerminalLog, options: ModalRunOptions) -> None:
    image = options.image or f"modal.Image.debian_slim(python_version='{options.python_version}')"
    gpu = options.gpu or "none"
    cpu = options.cpu if options.cpu is not None else "default"
    memory = f"{options.memory} MiB" if options.memory is not None else "default"
    log.info(f"Starting remote execution on Modal (image={image}, gpu={gpu}).")
    log.info(
        "Modal resources: "
        f"cpu={cpu}, memory={memory}, timeout={options.timeout}s, "
        f"kernel={options.kernel_name}, pip_packages={len(options.pip_packages)}, "
        f"apt_packages={len(options.apt_packages)}, secrets={len(options.secrets)}, "
        f"volumes={len(options.volumes)}, workdir={options.workdir}."
    )
    log.info("Modal setup and image preparation started.")
    log.info("Image setup can take a few minutes before cell progress appears.")


def _run_preflight(log: TerminalLog, output_path: Path, options: ModalRunOptions) -> None:
    _preflight_output_path(output_path)
    log.info(f"Output path is writable or creatable: {output_path}.")
    report = preflight_modal_run(options)
    for check in report.checks:
        log.info(f"Preflight: {check}")
    for warning in report.warnings:
        log.info(f"Preflight warning: {warning}")


def _preflight_output_path(output_path: Path) -> None:
    if output_path.exists() and output_path.is_dir():
        raise ModalSetupError(f"Output path is a directory: {output_path}")

    parent = output_path.parent
    existing_parent = parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.exists():
        raise ModalSetupError(f"No existing parent directory for output path: {output_path}")
    if not existing_parent.is_dir():
        raise ModalSetupError(f"Output parent is not a directory: {existing_parent}")

    probe = existing_parent / f".runbook-preflight-{uuid.uuid4().hex}"
    try:
        probe.write_text("", encoding="utf-8")
    except Exception as exc:
        raise ModalSetupError(f"Output parent is not writable: {existing_parent}: {exc}") from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def _print_requirements_plan(log: TerminalLog, result: RequirementsLoadResult) -> None:
    for line in requirements_summary_lines(result.requirements):
        log.info(f"Requirements: {line}")
    if result.previous_requirements is not None:
        diff_lines = requirements_diff_lines(
            result.previous_requirements,
            result.requirements,
        )
        if diff_lines:
            for line in diff_lines:
                log.info(f"Requirements diff: {line}")
        else:
            log.info("Requirements diff: no effective changes.")


def _print_cell_output(log: TerminalLog, event: dict[str, Any]) -> None:
    text = str(event.get("text", ""))
    if not text:
        return
    cell = event.get("cell", "?")
    total = event.get("total_cells", "?")
    stream = event.get("name")
    label = f"Cell {cell}/{total} output"
    if stream:
        label += f" ({stream})"
    log.info(f"{label}: {_one_line(text)}")


def _one_line(text: str, *, max_chars: int = 240) -> str:
    line = " ".join(text.strip().split())
    if len(line) <= max_chars:
        return line
    return f"{line[: max_chars - 3]}..."


def _notebook_status(
    *,
    finished: bool,
    failure: dict | None,
    startup_failure: dict | None,
) -> str:
    if failure is not None:
        return "failed"
    if startup_failure is not None:
        return "startup_failed"
    if finished:
        return "finished"
    return "partial"


def _attach_run_manifest(
    notebook_json: str,
    requirements_result: RequirementsLoadResult,
    options: ModalRunOptions,
    debug: dict | None,
    *,
    status: str,
    completed: int,
    total_cells: int,
) -> str:
    try:
        notebook = nbformat.reads(notebook_json, as_version=4)
    except Exception:
        return notebook_json

    notebook.metadata["runbook"] = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed_cells": completed,
        "total_cells": total_cells,
        "requirements_path": str(requirements_result.path),
        "requirements_generated": requirements_result.generated,
        "planner": {
            "provider": requirements_result.requirements.planner.provider,
            "model": requirements_result.requirements.planner.model,
            "confidence": requirements_result.requirements.planner.confidence,
            "notes": list(requirements_result.requirements.planner.notes),
            "source_hash": requirements_result.requirements.planner.source_hash,
        },
        "runtime": {
            "image": options.image,
            "gpu": options.gpu,
            "cpu": options.cpu,
            "memory": options.memory,
            "timeout": options.timeout,
            "kernel_name": options.kernel_name,
            "python_version": options.python_version,
            "build_toolchain": options.build_toolchain,
            "pip_index_url": options.pip_index_url,
            "pip_extra_index_urls": list(options.pip_extra_index_urls),
            "workdir": options.workdir,
        },
        "packages": {
            "pip": list(options.pip_packages),
            "apt": list(options.apt_packages),
        },
        "modal": {
            "secrets": list(options.secrets),
            "volumes": list(options.volumes),
            "debug": debug or {},
        },
    }
    return nbformat.writes(notebook)


def _load_or_prompt_openrouter_settings(console: Console) -> OpenRouterSettings:
    init_settings_dir()
    settings = load_openrouter_settings()
    if settings.api_key:
        return settings

    console.print(
        f"No OpenRouter settings found. Runbook stores them in {runbook_env_path()}."
    )
    api_key = typer.prompt(
        "OpenRouter API key",
        default="",
        hide_input=True,
        show_default=False,
    ).strip()
    if not api_key:
        return OpenRouterSettings(api_key=None, model=settings.model)

    model = typer.prompt(
        "OpenRouter model",
        default=settings.model,
    ).strip() or settings.model
    settings = OpenRouterSettings(api_key=api_key, model=model)
    save_openrouter_settings(settings)
    return settings


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _write_notebook_if_available(
    output_path: Path,
    notebook_data: str | None,
    log: TerminalLog | None = None,
    *,
    partial: bool = False,
) -> bool:
    if notebook_data is None:
        if log is not None:
            log.info("No notebook payload available to write.")
        return False
    started = perf_counter()
    atomic_write_text(output_path, notebook_data)
    if log is not None:
        label = "partial notebook" if partial else "notebook"
        size = _format_bytes(len(notebook_data.encode("utf-8")))
        log.info(
            f"Wrote {label} to {output_path} in "
            f"{_format_duration(perf_counter() - started)} ({size})."
        )
    return True


class TerminalLog:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._started = perf_counter()

    def info(self, message: str) -> None:
        self._console.print(f"T+{_format_duration(perf_counter() - self._started)} | {message}")

    def done(self, label: str, started: float) -> None:
        self.info(f"{label} in {_format_duration(perf_counter() - started)}.")


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


if __name__ == "__main__":
    app()
