from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn

Event = dict[str, Any]


class NotebookProgress:
    """Small wrapper around Rich progress for notebook execution."""

    def __init__(self, console: Console) -> None:
        self._progress = Progress(
            TextColumn("[bold]Cells"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[current]}"),
            console=console,
        )
        self._task_id: TaskID | None = None

    def __enter__(self) -> "NotebookProgress":
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def start(self, total_cells: int) -> None:
        self._task_id = self._progress.add_task(
            "cells",
            total=max(total_cells, 1),
            completed=0,
            current=f"0/{total_cells}",
        )
        if total_cells == 0:
            self.update(0, 0)

    def update(self, completed: int, total_cells: int) -> None:
        if self._task_id is None:
            self.start(total_cells)
        assert self._task_id is not None
        total = max(total_cells, 1)
        display_completed = total if total_cells == 0 else completed
        self._progress.update(
            self._task_id,
            completed=display_completed,
            total=total,
            current=f"{completed}/{total_cells}",
        )

    def current(self, cell: int, total_cells: int) -> None:
        if self._task_id is None:
            self.start(total_cells)
        assert self._task_id is not None
        self._progress.update(self._task_id, current=f"{cell}/{total_cells}")


def print_debug_info(console: Console, title: str, debug: dict[str, Any] | None) -> None:
    if not debug:
        console.print(f"{title}: Modal debug info unavailable")
        return

    parts = []
    for key in ("app_name", "function_name", "function_call_id", "input_id"):
        value = debug.get(key)
        if value:
            parts.append(f"{key}={value}")
    dashboard_url = debug.get("dashboard_url")
    if dashboard_url:
        parts.append(f"dashboard={dashboard_url}")
    console.print(f"{title}: " + " ".join(parts))


def print_success_summary(
    console: Console,
    output_path: Path,
    completed: int,
    total_cells: int,
    debug: dict[str, Any] | None,
    *,
    allowed_error_count: int = 0,
) -> None:
    suffix = f" with {allowed_error_count} allowed error(s)" if allowed_error_count else ""
    console.print(
        f"Completed {completed}/{total_cells} executable cell(s){suffix}. "
        f"Wrote {output_path}."
    )
    print_debug_info(console, "Modal run", debug)


def print_failure_summary(
    console: Console,
    output_path: Path | None,
    failure: Event,
    debug: dict[str, Any] | None,
) -> None:
    notebook_cell = failure.get("notebook_cell")
    cell = failure.get("cell")
    total = failure.get("total_cells")
    original = ""
    if notebook_cell and notebook_cell != cell:
        original = f" (notebook cell {notebook_cell})"
    console.print(
        f"Failed at executable cell {cell}/{total}{original}, "
        f"type={failure.get('cell_type', 'unknown')}."
    )
    preview = failure.get("source_preview")
    if preview:
        console.print(f"Source: {preview}")
    console.print(f"{failure.get('error_type')}: {failure.get('message')}")
    if output_path is not None:
        console.print(f"Wrote partial notebook to {output_path}.")
    print_debug_info(console, "Modal run", debug)


def print_startup_failure_summary(
    console: Console,
    output_path: Path | None,
    failure: Event,
    debug: dict[str, Any] | None,
) -> None:
    console.print(
        f"Notebook startup failed before cell execution: "
        f"{failure.get('error_type')}: {failure.get('message')}"
    )
    if output_path is not None:
        console.print(f"Wrote partial notebook to {output_path}.")
    print_debug_info(console, "Modal run", debug)
