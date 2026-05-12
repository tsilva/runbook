from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from runbook.convert import NotebookConversionError, default_output_path, read_notebook
from runbook.modal_app import ModalRunOptions, ModalSetupError, stream_remote_events
from runbook.progress import (
    NotebookProgress,
    print_debug_info,
    print_failure_summary,
    print_startup_failure_summary,
    print_success_summary,
)
from runbook.requirements_plan import (
    DEFAULT_KERNEL_NAME,
    DEFAULT_TIMEOUT,
    ModalRequirements,
    NotebookRequirements,
    PackageRequirements,
    RequirementsConfigError,
    RequirementsLoadResult,
    RuntimeRequirements,
    companion_requirements_path,
    load_or_generate_requirements,
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
) -> None:
    """Execute a notebook remotely on Modal and write an executed .ipynb."""

    console = Console(stderr=True)
    init_settings_dir()
    try:
        converted = read_notebook(input_path)
    except NotebookConversionError as exc:
        console.print(f"Notebook conversion failed: {exc}")
        raise typer.Exit(1) from exc

    try:
        requirements_result = _load_or_create_requirements(
            console,
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
        )
    except RequirementsConfigError as exc:
        console.print(f"Execution requirements failed: {exc}")
        raise typer.Exit(1) from exc

    if requirements_result.generated:
        console.print(f"Wrote execution requirements to {requirements_result.path}.")
    elif requirements_result.path.exists():
        console.print(f"Using execution requirements from {requirements_result.path}.")
    else:
        console.print("Using CLI-provided execution requirements.")

    output_path = (output or default_output_path(converted.source_path)).expanduser().resolve()
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
    )

    notebook_data: str | None = None
    debug: dict | None = None
    failure: dict | None = None
    startup_failure: dict | None = None
    completed = 0
    total_cells = 0
    allowed_error_count = 0
    finished = False

    try:
        with tempfile.TemporaryDirectory(prefix="runbook-"):
            with NotebookProgress(console) as progress:
                for event in stream_remote_events(converted.notebook_json, options):
                    kind = event.get("event")
                    if kind == "started":
                        total_cells = int(event.get("total_cells", 0))
                        debug = event.get("debug") or debug
                        print_debug_info(console, "Modal run started", debug)
                        progress.start(total_cells)
                    elif kind == "cell_started":
                        progress.current(
                            int(event.get("cell", 0)),
                            int(event.get("total_cells", total_cells)),
                        )
                    elif kind == "cell_finished":
                        completed = int(event.get("completed", completed))
                        total_cells = int(event.get("total_cells", total_cells))
                        progress.update(completed, total_cells)
                    elif kind == "cell_failed":
                        if event.get("allowed"):
                            allowed_error_count += 1
                        else:
                            failure = event
                    elif kind == "startup_failed":
                        startup_failure = event
                    elif kind == "finished":
                        finished = True
                        completed = int(event.get("completed", completed))
                        total_cells = int(event.get("total_cells", total_cells))
                        progress.update(completed, total_cells)
                    elif kind == "notebook":
                        if event.get("format") == "ipynb-json":
                            notebook_data = str(event.get("data", ""))
    except ModalSetupError as exc:
        _write_notebook_if_available(output_path, notebook_data)
        console.print(f"Modal setup/execution failed: {exc}")
        if debug:
            print_debug_info(console, "Modal run", debug)
        raise typer.Exit(1) from exc
    except Exception as exc:
        _write_notebook_if_available(output_path, notebook_data)
        console.print(f"Remote execution failed: {exc}")
        if debug:
            print_debug_info(console, "Modal run", debug)
        raise typer.Exit(1) from exc

    wrote_output = _write_notebook_if_available(output_path, notebook_data)
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
    )


def _load_or_create_requirements(
    console: Console,
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
) -> RequirementsLoadResult:
    requirements_path = companion_requirements_path(input_path)
    if requirements_path.exists():
        return load_or_generate_requirements(input_path, notebook_json)

    settings = _load_or_prompt_openrouter_settings(console)
    if settings.api_key:
        console.print(
            f"Generating execution requirements with OpenRouter ({settings.model})..."
        )
        return load_or_generate_requirements(
            input_path,
            notebook_json,
            api_key=settings.api_key,
            model=settings.model,
        )

    if not image:
        raise RequirementsConfigError(
            "No companion requirements file exists and OpenRouter settings were "
            "not provided. Pass --image to run without generating requirements, "
            "plus --gpu/--pip-package/--apt-package as needed."
        )

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


def _write_notebook_if_available(output_path: Path, notebook_data: str | None) -> bool:
    if notebook_data is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notebook_data, encoding="utf-8")
    return True


if __name__ == "__main__":
    app()
