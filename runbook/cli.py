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
        int,
        typer.Option("--timeout", help="Modal function and per-cell timeout in seconds."),
    ] = 3600,
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
    allow_errors: Annotated[
        bool,
        typer.Option("--allow-errors", help="Continue executing after cell errors."),
    ] = False,
    kernel_name: Annotated[
        str,
        typer.Option("--kernel-name", help="Jupyter kernel name."),
    ] = "python3",
) -> None:
    """Execute a notebook remotely on Modal and write an executed .ipynb."""

    console = Console(stderr=True)
    try:
        converted = read_notebook(input_path)
    except NotebookConversionError as exc:
        console.print(f"Notebook conversion failed: {exc}")
        raise typer.Exit(1) from exc

    output_path = (output or default_output_path(converted.source_path)).expanduser().resolve()
    options = ModalRunOptions(
        timeout=timeout,
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        secrets=secret or [],
        volumes=volume or [],
        image=image,
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


def _write_notebook_if_available(output_path: Path, notebook_data: str | None) -> bool:
    if notebook_data is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notebook_data, encoding="utf-8")
    return True


if __name__ == "__main__":
    app()
