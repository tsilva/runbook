from __future__ import annotations

import os
import traceback as traceback_module
from collections.abc import Iterator
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

Event = dict[str, Any]


def execute_notebook_events(
    notebook_json: str,
    *,
    allow_errors: bool,
    timeout: int,
    kernel_name: str,
    debug: dict[str, Any] | None = None,
    workdir: str = "/tmp/runbook",
) -> Iterator[Event]:
    """Execute notebook JSON one code cell at a time and yield progress events."""

    debug_info = _debug_info(debug)
    notebook = None

    try:
        notebook = nbformat.reads(notebook_json, as_version=4)
    except Exception as exc:
        yield _startup_failed_event(exc)
        return

    code_cell_indices = [
        index for index, cell in enumerate(notebook.cells) if cell.get("cell_type") == "code"
    ]
    total_cells = len(code_cell_indices)
    yield {
        "event": "started",
        "total_cells": total_cells,
        "debug": debug_info,
    }

    os.makedirs(workdir, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=allow_errors,
        resources={"metadata": {"path": workdir}},
    )

    completed = 0
    try:
        with client.setup_kernel(cwd=workdir):
            for executable_index, notebook_index in enumerate(code_cell_indices, start=1):
                cell = notebook.cells[notebook_index]
                yield {
                    "event": "cell_started",
                    "cell": executable_index,
                    "notebook_cell": notebook_index + 1,
                    "total_cells": total_cells,
                    "cell_type": cell.get("cell_type", "unknown"),
                    "source_preview": source_preview(cell.get("source", "")),
                }

                try:
                    client.execute_cell(
                        cell,
                        notebook_index,
                        execution_count=executable_index,
                        store_history=True,
                    )
                except Exception as exc:
                    if not isinstance(exc, CellExecutionError):
                        error = _exception_error(exc)
                    else:
                        error = _cell_error(cell)
                        if not error["error_type"]:
                            error = _exception_error(exc)
                    yield {
                        "event": "cell_failed",
                        "cell": executable_index,
                        "notebook_cell": notebook_index + 1,
                        "total_cells": total_cells,
                        "cell_type": cell.get("cell_type", "unknown"),
                        "error_type": error["error_type"],
                        "message": error["message"],
                        "traceback": error["traceback"],
                        "source_preview": source_preview(cell.get("source", "")),
                        "allowed": False,
                    }
                    yield _notebook_event(notebook)
                    return

                completed += 1
                error = _cell_error(cell)
                if error["error_type"]:
                    yield {
                        "event": "cell_failed",
                        "cell": executable_index,
                        "notebook_cell": notebook_index + 1,
                        "total_cells": total_cells,
                        "cell_type": cell.get("cell_type", "unknown"),
                        "error_type": error["error_type"],
                        "message": error["message"],
                        "traceback": error["traceback"],
                        "source_preview": source_preview(cell.get("source", "")),
                        "allowed": True,
                    }

                yield {
                    "event": "cell_finished",
                    "cell": executable_index,
                    "notebook_cell": notebook_index + 1,
                    "completed": completed,
                    "total_cells": total_cells,
                    "status": "error" if error["error_type"] else "ok",
                }
    except Exception as exc:
        yield _startup_failed_event(exc)
        if notebook is not None:
            yield _notebook_event(notebook)
        return

    yield {"event": "finished", "completed": completed, "total_cells": total_cells}
    yield _notebook_event(notebook)


def source_preview(source: str, *, max_chars: int = 120) -> str:
    """Return a compact one-line cell source preview."""

    preview = " ".join(source.strip().split())
    if len(preview) <= max_chars:
        return preview
    return f"{preview[: max_chars - 3]}..."


def _notebook_event(notebook: nbformat.NotebookNode) -> Event:
    return {
        "event": "notebook",
        "format": "ipynb-json",
        "data": nbformat.writes(notebook),
    }


def _cell_error(cell: nbformat.NotebookNode) -> dict[str, str]:
    for output in reversed(cell.get("outputs", [])):
        if output.get("output_type") == "error":
            traceback_value = output.get("traceback", [])
            if isinstance(traceback_value, list):
                traceback_text = "\n".join(str(line) for line in traceback_value)
            else:
                traceback_text = str(traceback_value)
            return {
                "error_type": str(output.get("ename", "")),
                "message": str(output.get("evalue", "")),
                "traceback": traceback_text,
            }
    return {"error_type": "", "message": "", "traceback": ""}


def _startup_failed_event(exc: BaseException) -> Event:
    return {
        "event": "startup_failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": _format_exception(exc),
    }


def _exception_error(exc: BaseException) -> dict[str, str]:
    cell_error = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": _format_exception(exc),
    }
    return cell_error


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback_module.format_exception(exc))


def _debug_info(debug: dict[str, Any] | None) -> dict[str, Any]:
    info = dict(debug or {})
    try:
        import modal

        info.setdefault("function_call_id", modal.current_function_call_id())
        info.setdefault("input_id", modal.current_input_id())
    except Exception:
        info.setdefault("function_call_id", None)
        info.setdefault("input_id", None)
    return info
