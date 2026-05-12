from __future__ import annotations

import json
import shlex
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

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
    print_cell_output_panel,
    print_debug_info,
    print_failure_summary,
    print_key_value_panel,
    print_requirements_table,
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
OutputMode = Literal["modern", "verbose", "plain", "jsonl"]


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(help="Input .ipynb or Jupytext .py file.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output notebook base path."),
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
            "--generate-requirements",
            "--regenerate-requirements",
            help="Generate the companion requirements file with the configured LLM.",
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Use the detailed timestamped event stream."),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable Rich panels/progress and print plain text."),
    ] = False,
    jsonl: Annotated[
        bool,
        typer.Option("--jsonl", help="Emit machine-readable JSON Lines output."),
    ] = False,
) -> None:
    """Execute a notebook remotely on Modal and write .running/.finished notebooks."""

    mode = _resolve_output_mode(verbose=verbose, plain=plain, jsonl=jsonl)
    console = Console(stderr=True, no_color=mode in {"plain", "jsonl"}, soft_wrap=mode == "jsonl")
    log = TerminalLog(console, mode=mode)
    _print_run_header(console, input_path, mode=mode)
    log.info("Runbook run started.", event="run_started")

    settings_started = perf_counter()
    init_settings_dir()
    log.done("Settings directory initialized", settings_started)

    conversion_started = perf_counter()
    log.info(f"Reading notebook from {input_path}.")
    try:
        converted = read_notebook(input_path)
    except NotebookConversionError as exc:
        _print_error(console, f"Notebook conversion failed: {exc}", mode=mode)
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
        _print_error(console, f"Execution requirements failed: {exc}", mode=mode)
        raise typer.Exit(1) from exc
    log.done("Execution requirements resolved", requirements_started)

    if requirements_result.generated:
        log.info(f"Wrote execution requirements to {requirements_result.path}.")
    elif requirements_result.path.exists():
        log.info(f"Using execution requirements from {requirements_result.path}.")
    else:
        log.info("Using CLI-provided execution requirements.")
    _print_requirements_plan(console, log, requirements_result, mode=mode)

    requested_output_path = (
        output or default_output_path(converted.source_path)
    ).expanduser().resolve()
    running_output_path, finished_output_path = _run_output_paths(requested_output_path)
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
        _run_preflight(log, [running_output_path, finished_output_path], options)
    except ModalSetupError as exc:
        log.done("Preflight failed", preflight_started)
        _print_error(console, f"Preflight failed: {exc}", mode=mode)
        raise typer.Exit(1) from exc
    log.done("Preflight completed", preflight_started)

    if dry_run:
        log.info("Dry run complete; remote execution skipped.", event="dry_run_complete")
        return

    _print_remote_execution_start(console, log, options, mode=mode)

    live_notebook = LiveNotebookWriter(
        converted.notebook_json,
        running_output_path,
        final_output_path=finished_output_path,
        requirements_result=requirements_result,
        options=options,
    )
    live_notebook.write(status="pending", completed=0, total_cells=0, debug=None)
    log.info(
        f"Initialized live output notebook at {running_output_path}.",
        event="live_notebook_initialized",
        output_path=str(running_output_path),
    )

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
            with NotebookProgress(console, enabled=mode == "modern") as progress:
                for event in stream_remote_events(converted.notebook_json, options):
                    event_data: dict[str, Any] = dict(event)
                    kind = event_data.get("event")
                    if not modal_setup_reported:
                        log.done(
                            "Modal setup and image preparation",
                            modal_setup_started,
                            event="modal_setup_finished",
                        )
                        modal_setup_reported = True
                    if kind == "started":
                        remote_execution_started = perf_counter()
                        total_cells = int(event_data.get("total_cells", 0))
                        event_debug = event_data.get("debug")
                        debug = event_debug if isinstance(event_debug, dict) else debug
                        live_notebook.write(
                            status="running",
                            completed=0,
                            total_cells=total_cells,
                            debug=debug,
                        )
                        print_debug_info(console, "Modal run started", debug, mode=mode)
                        log.event("modal_started", total_cells=total_cells, debug=debug or {})
                        log.info(
                            "Remote notebook execution started with "
                            f"{total_cells} executable cell(s).",
                            event="remote_execution_started",
                            total_cells=total_cells,
                        )
                        progress.start(total_cells)
                    elif kind == "cell_started":
                        cell = int(event_data.get("cell", 0))
                        event_total = int(event_data.get("total_cells", total_cells))
                        cell_started_at[cell] = perf_counter()
                        source_preview = str(event_data.get("source_preview") or "").strip()
                        live_notebook.cell_started(event_data, completed=completed, debug=debug)
                        log.info(
                            f"Cell {cell}/{event_total} started.",
                            event="cell_started",
                            cell=cell,
                            total_cells=event_total,
                            notebook_cell=event_data.get("notebook_cell"),
                            source_preview=source_preview,
                        )
                        progress.current(
                            cell,
                            event_total,
                            status=_one_line(source_preview, max_chars=48),
                        )
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
                        log.info(
                            f"Cell {cell}/{total_cells} finished{elapsed}.",
                            event="cell_finished",
                            cell=cell,
                            total_cells=total_cells,
                            completed=completed,
                            elapsed=elapsed.strip(),
                        )
                        live_notebook.cell_finished(
                            event_data,
                            status=str(event_data.get("status", "ok")),
                            debug=debug,
                        )
                        progress.update(completed, total_cells, status="ok")
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
                            f"{event_data.get('error_type')}: {event_data.get('message')}",
                            event="cell_failed",
                            cell=cell,
                            total_cells=event_total,
                            error_type=event_data.get("error_type"),
                            error_message=event_data.get("message"),
                            allowed=bool(event_data.get("allowed")),
                        )
                        if event_data.get("allowed"):
                            allowed_error_count += 1
                        else:
                            failure = event_data
                        live_notebook.cell_failed(
                            event_data,
                            completed=completed,
                            debug=debug,
                        )
                    elif kind == "cell_output":
                        live_notebook.cell_output(event_data, completed=completed, debug=debug)
                        _print_cell_output(console, log, event_data, mode=mode)
                    elif kind == "startup_failed":
                        startup_failure = event_data
                        live_notebook.startup_failed(event_data, debug=debug)
                        log.info(
                            "Notebook startup failed before cell execution: "
                            f"{event_data.get('error_type')}: {event_data.get('message')}",
                            event="startup_failed",
                            error_type=event_data.get("error_type"),
                            error_message=event_data.get("message"),
                        )
                    elif kind == "finished":
                        finished = True
                        completed = int(event_data.get("completed", completed))
                        total_cells = int(event_data.get("total_cells", total_cells))
                        if remote_execution_started is not None:
                            log.done(
                                "Remote notebook execution",
                                remote_execution_started,
                                event="remote_execution_finished",
                            )
                        progress.update(completed, total_cells, status="complete")
                        live_notebook.write(
                            status="finished",
                            completed=completed,
                            total_cells=total_cells,
                            debug=debug,
                        )
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
                            live_notebook.replace_with_final(notebook_data)
                            log.info(
                                "Received output notebook payload "
                                f"({_format_bytes(len(notebook_data.encode('utf-8')))}).",
                                event="notebook_payload_received",
                                bytes=len(notebook_data.encode("utf-8")),
                            )
    except ModalSetupError as exc:
        if not modal_setup_reported:
            log.done("Modal setup failed", modal_setup_started, event="modal_setup_failed")
        if notebook_data is None:
            live_notebook.finalize_current(
                status="failed",
                completed=completed,
                total_cells=total_cells,
                debug=debug,
            )
        else:
            _write_notebook_if_available(finished_output_path, notebook_data, log, partial=True)
            _delete_running_output(running_output_path, log)
        _print_error(console, f"Modal setup/execution failed: {exc}", mode=mode)
        if debug:
            print_debug_info(console, "Modal run", debug, mode=mode)
        raise typer.Exit(1) from exc
    except Exception as exc:
        if not modal_setup_reported:
            log.done(
                "Remote execution failed before first Modal event",
                modal_setup_started,
                event="remote_execution_failed_before_event",
            )
        if notebook_data is None:
            live_notebook.finalize_current(
                status="failed",
                completed=completed,
                total_cells=total_cells,
                debug=debug,
            )
        else:
            _write_notebook_if_available(finished_output_path, notebook_data, log, partial=True)
            _delete_running_output(running_output_path, log)
        _print_error(console, f"Remote execution failed: {exc}", mode=mode)
        if debug:
            print_debug_info(console, "Modal run", debug, mode=mode)
        raise typer.Exit(1) from exc

    if not modal_setup_reported:
        log.done(
            "Modal setup and image preparation ended without events",
            modal_setup_started,
            event="modal_setup_ended_without_events",
        )

    if notebook_data is None:
        live_notebook.finalize_current(
            status=_notebook_status(
                finished=finished,
                failure=failure,
                startup_failure=startup_failure,
            ),
            completed=completed,
            total_cells=total_cells,
            debug=debug,
        )
    wrote_output = _write_notebook_if_available(finished_output_path, notebook_data, log)
    if wrote_output:
        _delete_running_output(running_output_path, log)
    if not wrote_output and finished_output_path.exists():
        wrote_output = True
    written_path = finished_output_path if wrote_output else None

    if startup_failure is not None:
        print_startup_failure_summary(console, written_path, startup_failure, debug, mode=mode)
        raise typer.Exit(1)

    if failure is not None:
        print_failure_summary(console, written_path, failure, debug, mode=mode)
        raise typer.Exit(1)

    if not finished:
        _print_error(
            console,
            "Remote execution ended before a finished event was received.",
            mode=mode,
        )
        if written_path is not None:
            log.info(f"Wrote partial notebook to {written_path}.", event="partial_notebook_written")
        if debug:
            print_debug_info(console, "Modal run", debug, mode=mode)
        raise typer.Exit(1)

    if written_path is None:
        _print_error(
            console,
            "Remote execution finished without returning an output notebook.",
            mode=mode,
        )
        raise typer.Exit(1)

    print_success_summary(
        console,
        written_path,
        completed,
        total_cells,
        debug,
        allowed_error_count=allowed_error_count,
        mode=mode,
    )
    log.event(
        "run_finished",
        status="finished",
        output_path=str(written_path),
        completed=completed,
        total_cells=total_cells,
        allowed_error_count=allowed_error_count,
        debug=debug or {},
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

    if not regenerate_requirements:
        if not image:
            raise RequirementsConfigError(
                _missing_requirements_message(input_path, requirements_path)
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

    raise RequirementsConfigError(
        "OpenRouter settings were not provided, so Runbook could not generate "
        f"{requirements_path}. Pass --image and any needed package flags to run "
        "without a companion requirements file."
    )


def _missing_requirements_message(input_path: Path, requirements_path: Path) -> str:
    command = (
        f"runbook {shlex.quote(str(input_path))} --generate-requirements --dry-run"
    )
    return (
        f"Companion requirements file does not exist: {requirements_path}\n"
        "Runbook will not call the LLM automatically. To generate it with the "
        f"configured LLM, run:\n\n  {command}\n\n"
        "To run without a companion requirements file, pass --image plus any "
        "needed --gpu/--pip-package/--apt-package flags."
    )


def _print_remote_execution_start(
    console: Console,
    log: TerminalLog,
    options: ModalRunOptions,
    *,
    mode: OutputMode,
) -> None:
    image = options.image or f"modal.Image.debian_slim(python_version='{options.python_version}')"
    gpu = options.gpu or "none"
    cpu = options.cpu if options.cpu is not None else "default"
    memory = f"{options.memory} MiB" if options.memory is not None else "default"
    if mode == "modern":
        print_key_value_panel(
            console,
            "[cyan]Modal run[/cyan]",
            [
                ("Image", image),
                ("GPU", gpu),
                ("CPU", str(cpu)),
                ("Memory", str(memory)),
                ("Timeout", f"{options.timeout}s"),
                ("Kernel", options.kernel_name),
                ("Packages", f"{len(options.pip_packages)} pip, {len(options.apt_packages)} apt"),
                ("Build", "toolchain enabled" if options.build_toolchain else "no toolchain"),
                ("Workdir", options.workdir),
            ],
        )
    log.info(
        f"Starting remote execution on Modal (image={image}, gpu={gpu}).",
        event="modal_execution_starting",
        image=image,
        gpu=gpu,
    )
    log.info(
        "Modal resources: "
        f"cpu={cpu}, memory={memory}, timeout={options.timeout}s, "
        f"kernel={options.kernel_name}, pip_packages={len(options.pip_packages)}, "
        f"apt_packages={len(options.apt_packages)}, secrets={len(options.secrets)}, "
        f"volumes={len(options.volumes)}, workdir={options.workdir}.",
        event="modal_resources",
        cpu=cpu,
        memory=memory,
        timeout=options.timeout,
        kernel=options.kernel_name,
        pip_packages=len(options.pip_packages),
        apt_packages=len(options.apt_packages),
        secrets=len(options.secrets),
        volumes=len(options.volumes),
        workdir=options.workdir,
    )
    log.info("Modal setup and image preparation started.", event="modal_setup_started")
    log.info(
        "Image setup can take a few minutes before cell progress appears.",
        event="modal_setup_hint",
    )


def _run_preflight(log: TerminalLog, output_paths: list[Path], options: ModalRunOptions) -> None:
    for output_path in output_paths:
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


def _print_requirements_plan(
    console: Console,
    log: TerminalLog,
    result: RequirementsLoadResult,
    *,
    mode: OutputMode,
) -> None:
    requirements = result.requirements
    runtime = requirements.runtime
    packages = requirements.packages
    planner = requirements.planner
    image = runtime.image or f"modal.Image.debian_slim(python_version='{runtime.python_version}')"
    gpu = runtime.gpu or "none"
    cpu = runtime.cpu if runtime.cpu is not None else "default"
    memory = f"{runtime.memory} MiB" if runtime.memory is not None else "default"
    confidence = f"{planner.confidence:.2f}" if planner.confidence is not None else "unknown"
    if mode == "modern":
        print_requirements_table(
            console,
            [
                ("Planner", f"{planner.provider} / {planner.model} / confidence {confidence}"),
                ("Runtime", f"{image} / GPU {gpu} / CPU {cpu} / memory {memory}"),
                ("Timeout", f"{runtime.timeout}s"),
                ("Kernel", runtime.kernel_name),
                ("Packages", f"{len(packages.pip)} pip, {len(packages.apt)} apt"),
                ("Build", "toolchain enabled" if runtime.build_toolchain else "no toolchain"),
                ("Secrets", str(len(requirements.modal.secrets))),
                ("Volumes", str(len(requirements.modal.volumes))),
            ],
        )
    for line in requirements_summary_lines(result.requirements):
        log.info(f"Requirements: {line}", event="requirements_summary", summary=line)
    if result.previous_requirements is not None:
        diff_lines = requirements_diff_lines(
            result.previous_requirements,
            result.requirements,
        )
        if diff_lines:
            for line in diff_lines:
                log.info(f"Requirements diff: {line}", event="requirements_diff", diff=line)
        else:
            log.info(
                "Requirements diff: no effective changes.",
                event="requirements_diff",
                diff="no effective changes",
            )


def _print_cell_output(
    console: Console,
    log: TerminalLog,
    event: dict[str, Any],
    *,
    mode: OutputMode,
) -> None:
    text = str(event.get("text", ""))
    if not text:
        return
    cell = event.get("cell", "?")
    total = event.get("total_cells", "?")
    stream = event.get("name")
    label = f"Cell {cell}/{total} output"
    if stream:
        label += f" ({stream})"
    preview = _one_line(text)
    log.info(
        f"{label}: {preview}",
        event="cell_output",
        cell=cell,
        total_cells=total,
        stream=stream,
        text=preview,
    )
    if mode == "modern":
        print_cell_output_panel(
            console,
            cell_label=f"Cell {cell}/{total}",
            stream=str(stream) if stream else None,
            text=preview,
        )


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


class LiveNotebookWriter:
    """Maintains the output notebook on disk while remote execution is still running."""

    def __init__(
        self,
        notebook_json: str,
        running_output_path: Path,
        *,
        final_output_path: Path,
        requirements_result: RequirementsLoadResult,
        options: ModalRunOptions,
    ) -> None:
        self._notebook = nbformat.reads(notebook_json, as_version=4)
        self._running_output_path = running_output_path
        self._final_output_path = final_output_path
        self._requirements_result = requirements_result
        self._options = options
        self._clear_outputs()

    def write(
        self,
        *,
        status: str,
        completed: int,
        total_cells: int,
        debug: dict | None,
        current_cell: int | None = None,
        current_notebook_cell: int | None = None,
    ) -> None:
        self._set_metadata(
            status=status,
            completed=completed,
            total_cells=total_cells,
            debug=debug,
            current_cell=current_cell,
            current_notebook_cell=current_notebook_cell,
        )
        atomic_write_text(self._running_output_path, nbformat.writes(self._notebook))

    def cell_started(
        self,
        event: dict[str, Any],
        *,
        completed: int,
        debug: dict | None,
    ) -> None:
        cell_index = _event_notebook_index(event)
        if cell_index is None:
            return
        cell = self._notebook.cells[cell_index]
        cell.metadata["runbook"] = {
            "execution_state": "running",
            "executable_cell": event.get("cell"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_preview": event.get("source_preview", ""),
        }
        if cell.get("cell_type") == "code":
            cell["execution_count"] = event.get("cell")
        self.write(
            status="running",
            completed=completed,
            total_cells=int(event.get("total_cells", 0)),
            debug=debug,
            current_cell=_optional_event_int(event, "cell"),
            current_notebook_cell=_optional_event_int(event, "notebook_cell"),
        )

    def cell_output(
        self,
        event: dict[str, Any],
        *,
        completed: int,
        debug: dict | None,
    ) -> None:
        cell_index = _event_notebook_index(event)
        if cell_index is None:
            return
        output = event.get("output")
        if isinstance(output, dict):
            self._notebook.cells[cell_index].outputs.append(nbformat.from_dict(output))
        else:
            self._notebook.cells[cell_index].outputs.append(
                _synthetic_output_from_event(event)
            )
        self.write(
            status="running",
            completed=completed,
            total_cells=int(event.get("total_cells", 0)),
            debug=debug,
            current_cell=_optional_event_int(event, "cell"),
            current_notebook_cell=_optional_event_int(event, "notebook_cell"),
        )

    def cell_failed(
        self,
        event: dict[str, Any],
        *,
        completed: int,
        debug: dict | None,
    ) -> None:
        cell_index = _event_notebook_index(event)
        if cell_index is None:
            return
        cell = self._notebook.cells[cell_index]
        cell.metadata["runbook"] = {
            **dict(cell.metadata.get("runbook", {})),
            "execution_state": "allowed_error" if event.get("allowed") else "failed",
            "error_type": event.get("error_type"),
            "message": event.get("message"),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if not _cell_has_error_output(cell, event):
            cell.outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=str(event.get("error_type", "")),
                    evalue=str(event.get("message", "")),
                    traceback=str(event.get("traceback", "")).splitlines(),
                )
            )
        self.write(
            status="running" if event.get("allowed") else "failed",
            completed=completed,
            total_cells=int(event.get("total_cells", 0)),
            debug=debug,
            current_cell=_optional_event_int(event, "cell"),
            current_notebook_cell=_optional_event_int(event, "notebook_cell"),
        )

    def cell_finished(
        self,
        event: dict[str, Any],
        *,
        status: str,
        debug: dict | None,
    ) -> None:
        cell_index = _event_notebook_index(event)
        if cell_index is not None:
            cell = self._notebook.cells[cell_index]
            cell.metadata["runbook"] = {
                **dict(cell.metadata.get("runbook", {})),
                "execution_state": "error" if status == "error" else "finished",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        self.write(
            status="running",
            completed=int(event.get("completed", 0)),
            total_cells=int(event.get("total_cells", 0)),
            debug=debug,
            current_cell=None,
            current_notebook_cell=None,
        )

    def startup_failed(self, event: dict[str, Any], *, debug: dict | None) -> None:
        self._notebook.metadata["runbook_startup_error"] = {
            "error_type": event.get("error_type"),
            "message": event.get("message"),
            "traceback": event.get("traceback"),
        }
        self.write(status="startup_failed", completed=0, total_cells=0, debug=debug)

    def replace_with_final(self, notebook_json: str) -> None:
        self._notebook = nbformat.reads(notebook_json, as_version=4)
        atomic_write_text(self._final_output_path, notebook_json)
        try:
            self._running_output_path.unlink()
        except FileNotFoundError:
            pass

    def finalize_current(
        self,
        *,
        status: str,
        completed: int,
        total_cells: int,
        debug: dict | None,
    ) -> None:
        self._set_metadata(
            status=status,
            completed=completed,
            total_cells=total_cells,
            debug=debug,
            current_cell=None,
            current_notebook_cell=None,
        )
        atomic_write_text(self._final_output_path, nbformat.writes(self._notebook))
        try:
            self._running_output_path.unlink()
        except FileNotFoundError:
            pass

    def _clear_outputs(self) -> None:
        for index, cell in enumerate(self._notebook.cells, start=1):
            if cell.get("cell_type") != "code":
                continue
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.metadata["runbook"] = {
                "execution_state": "pending",
                "notebook_cell": index,
            }

    def _set_metadata(
        self,
        *,
        status: str,
        completed: int,
        total_cells: int,
        debug: dict | None,
        current_cell: int | None,
        current_notebook_cell: int | None,
    ) -> None:
        requirements = self._requirements_result.requirements
        self._notebook.metadata["runbook"] = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "live_update": True,
            "current_cell": current_cell,
            "current_notebook_cell": current_notebook_cell,
            "completed_cells": completed,
            "total_cells": total_cells,
            "requirements_path": str(self._requirements_result.path),
            "requirements_generated": self._requirements_result.generated,
            "planner": {
                "provider": requirements.planner.provider,
                "model": requirements.planner.model,
                "confidence": requirements.planner.confidence,
                "notes": list(requirements.planner.notes),
                "source_hash": requirements.planner.source_hash,
            },
            "runtime": {
                "image": self._options.image,
                "gpu": self._options.gpu,
                "cpu": self._options.cpu,
                "memory": self._options.memory,
                "timeout": self._options.timeout,
                "kernel_name": self._options.kernel_name,
                "python_version": self._options.python_version,
                "build_toolchain": self._options.build_toolchain,
                "pip_index_url": self._options.pip_index_url,
                "pip_extra_index_urls": list(self._options.pip_extra_index_urls),
                "workdir": self._options.workdir,
            },
            "packages": {
                "pip": list(self._options.pip_packages),
                "apt": list(self._options.apt_packages),
            },
            "modal": {
                "secrets": list(self._options.secrets),
                "volumes": list(self._options.volumes),
                "debug": debug or {},
            },
        }


def _event_notebook_index(event: dict[str, Any]) -> int | None:
    notebook_cell = _optional_event_int(event, "notebook_cell")
    if notebook_cell is None or notebook_cell < 1:
        return None
    return notebook_cell - 1


def _optional_event_int(event: dict[str, Any], key: str) -> int | None:
    value = event.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _synthetic_output_from_event(event: dict[str, Any]) -> nbformat.NotebookNode:
    output_type = str(event.get("output_type", "stream"))
    text = str(event.get("text", ""))
    if output_type == "error":
        return nbformat.v4.new_output("error", ename="", evalue=text, traceback=[text])
    if output_type in {"display_data", "execute_result"}:
        return nbformat.v4.new_output(output_type, data={"text/plain": text}, metadata={})
    return nbformat.v4.new_output(
        "stream",
        name=str(event.get("name", "stdout") or "stdout"),
        text=text,
    )


def _cell_has_error_output(cell: nbformat.NotebookNode, event: dict[str, Any]) -> bool:
    for output in cell.get("outputs", []):
        if output.get("output_type") != "error":
            continue
        if str(output.get("ename", "")) == str(event.get("error_type", "")):
            return True
    return False


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


def _run_output_paths(requested_output_path: Path) -> tuple[Path, Path]:
    base = _output_base_path(requested_output_path)
    return (
        base.with_name(f"{base.name}.running.ipynb"),
        base.with_name(f"{base.name}.finished.ipynb"),
    )


def _output_base_path(path: Path) -> Path:
    name = path.name
    for suffix in (".running.ipynb", ".finished.ipynb", ".executed.ipynb", ".ipynb"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path.with_suffix("")


def _delete_running_output(path: Path, log: TerminalLog | None = None) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    if log is not None:
        log.info(f"Deleted live output notebook {path}.", event="live_notebook_deleted")


class TerminalLog:
    def __init__(self, console: Console, *, mode: OutputMode) -> None:
        self._console = console
        self._mode = mode
        self._started = perf_counter()

    def info(self, message: str, *, event: str = "log", **fields: Any) -> None:
        elapsed = perf_counter() - self._started
        if self._mode == "jsonl":
            self.event(event, message=message, elapsed_ms=round(elapsed * 1000), **fields)
            return
        if self._mode == "modern":
            self._console.print(
                f"[dim]T+{_format_duration(elapsed)}[/dim] [cyan]›[/cyan] {message}"
            )
            return
        self._console.print(f"T+{_format_duration(elapsed)} | {message}")

    def done(self, label: str, started: float, *, event: str = "step_finished") -> None:
        duration = perf_counter() - started
        self.info(
            f"{label} in {_format_duration(duration)}.",
            event=event,
            label=label,
            duration_ms=round(duration * 1000),
        )

    def event(self, event: str, **fields: Any) -> None:
        if self._mode != "jsonl":
            return
        payload = {"event": event, **fields}
        self._console.print(
            json.dumps(payload, sort_keys=True, default=str),
            highlight=False,
            soft_wrap=True,
        )


def _resolve_output_mode(*, verbose: bool, plain: bool, jsonl: bool) -> OutputMode:
    selected = [verbose, plain, jsonl]
    if sum(bool(value) for value in selected) > 1:
        raise typer.BadParameter("Choose only one output mode: --verbose, --plain, or --jsonl.")
    if jsonl:
        return "jsonl"
    if plain:
        return "plain"
    if verbose:
        return "verbose"
    return "modern"


def _print_run_header(console: Console, input_path: Path, *, mode: OutputMode) -> None:
    if mode == "modern":
        print_key_value_panel(
            console,
            "[bold cyan]Runbook[/bold cyan]",
            [("Notebook", str(input_path)), ("Mode", "remote Modal execution")],
            border_style="cyan",
        )


def _print_error(console: Console, message: str, *, mode: OutputMode) -> None:
    if mode == "jsonl":
        console.print(
            json.dumps({"event": "error", "message": message}),
            highlight=False,
            soft_wrap=True,
        )
    elif mode == "modern":
        console.print(f"[red]{message}[/red]")
    else:
        console.print(message)


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
