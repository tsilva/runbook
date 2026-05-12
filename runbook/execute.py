from __future__ import annotations

import os
import queue
import threading
import traceback as traceback_module
from collections.abc import Iterator
from typing import Any, cast

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from runbook.events import DebugInfo, Event


class StreamingNotebookClient(NotebookClient):
    """NotebookClient that reports newly appended outputs through a callback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.runbook_output_callback = kwargs.pop("runbook_output_callback", None)
        super().__init__(*args, **kwargs)

    def output(
        self,
        outs: list[nbformat.NotebookNode],
        msg: dict[str, Any],
        display_id: str | None,
        cell_index: int,
    ) -> nbformat.NotebookNode | None:
        out = super().output(outs, msg, display_id, cell_index)
        if out is not None and self.runbook_output_callback is not None:
            self.runbook_output_callback(out)
        return out


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
    client = StreamingNotebookClient(
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
                    yield from _execute_cell_with_output_events(
                        client,
                        cell,
                        notebook_index,
                        executable_index=executable_index,
                        total_cells=total_cells,
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


def _debug_info(debug: dict[str, Any] | None) -> DebugInfo:
    info = dict(debug or {})
    try:
        import modal

        info.setdefault("function_call_id", modal.current_function_call_id())
        info.setdefault("input_id", modal.current_input_id())
    except Exception:
        info.setdefault("function_call_id", None)
        info.setdefault("input_id", None)
    return cast(DebugInfo, info)


def _execute_cell_with_output_events(
    client: StreamingNotebookClient,
    cell: nbformat.NotebookNode,
    notebook_index: int,
    *,
    executable_index: int,
    total_cells: int,
) -> Iterator[Event]:
    event_queue: queue.Queue[Event | BaseException | object] = queue.Queue()
    done = object()

    def output_callback(output: nbformat.NotebookNode) -> None:
        event = _cell_output_event(output, executable_index, notebook_index, total_cells)
        if event is not None:
            event_queue.put(event)

    def run_cell() -> None:
        client.runbook_output_callback = output_callback
        try:
            client.execute_cell(
                cell,
                notebook_index,
                execution_count=executable_index,
                store_history=True,
            )
        except BaseException as exc:
            event_queue.put(exc)
        finally:
            client.runbook_output_callback = None
            event_queue.put(done)

    thread = threading.Thread(target=run_cell, daemon=True)
    thread.start()
    error: BaseException | None = None

    while True:
        item = event_queue.get()
        if item is done:
            break
        if isinstance(item, BaseException):
            error = item
            continue
        yield cast(Event, item)

    thread.join()
    if error is not None:
        raise error


def _cell_output_event(
    output: nbformat.NotebookNode,
    executable_index: int,
    notebook_index: int,
    total_cells: int,
) -> Event | None:
    output_type = str(output.get("output_type", ""))
    event: dict[str, Any] = {
        "event": "cell_output",
        "cell": executable_index,
        "notebook_cell": notebook_index + 1,
        "total_cells": total_cells,
        "output_type": output_type,
        "output": _plain_json(output),
    }

    if output_type == "stream":
        event["name"] = str(output.get("name", ""))
        event["text"] = _truncate_output_text(str(output.get("text", "")))
    elif output_type in {"display_data", "execute_result"}:
        data = output.get("data", {})
        if isinstance(data, dict) and "text/plain" in data:
            event["text"] = _truncate_output_text(str(data["text/plain"]))
    elif output_type == "error":
        event["text"] = _truncate_output_text(
            f"{output.get('ename', '')}: {output.get('evalue', '')}"
        )

    if "text" not in event:
        return None
    return cast(Event, event)


def _plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _truncate_output_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
