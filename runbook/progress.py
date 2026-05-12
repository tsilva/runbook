from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

Event = dict[str, Any]
OutputMode = str


class NotebookProgress:
    """Small wrapper around Rich progress for notebook execution."""

    def __init__(self, console: Console, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._progress = Progress(
            TextColumn("[bold cyan]Cells"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[current]}"),
            TextColumn("[dim]{task.fields[status]}"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id: TaskID | None = None

    def __enter__(self) -> NotebookProgress:
        if self._enabled:
            self._progress.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._enabled:
            self._progress.__exit__(exc_type, exc, tb)

    def start(self, total_cells: int) -> None:
        if not self._enabled:
            return
        self._task_id = self._progress.add_task(
            "cells",
            total=max(total_cells, 1),
            completed=0,
            current=f"0/{total_cells}",
            status="waiting",
        )
        if total_cells == 0:
            self.update(0, 0)

    def update(self, completed: int, total_cells: int, *, status: str = "") -> None:
        if not self._enabled:
            return
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
            status=status,
        )

    def current(self, cell: int, total_cells: int, *, status: str = "running") -> None:
        if not self._enabled:
            return
        if self._task_id is None:
            self.start(total_cells)
        assert self._task_id is not None
        self._progress.update(self._task_id, current=f"{cell}/{total_cells}", status=status)


def print_debug_info(
    console: Console,
    title: str,
    debug: dict[str, Any] | None,
    *,
    mode: OutputMode = "modern",
) -> None:
    if mode == "jsonl":
        return
    if not debug:
        message = f"{title}: Modal debug info unavailable"
        console.print(f"[dim]{message}[/dim]" if mode == "modern" else message)
        return

    parts = []
    for key in ("app_name", "function_name", "function_call_id", "input_id"):
        value = debug.get(key)
        if value:
            parts.append(f"{key}={value}")
    dashboard_url = debug.get("dashboard_url")
    if dashboard_url:
        parts.append(f"dashboard={dashboard_url}")
    if mode == "modern":
        console.print(f"[dim]{title}[/dim] [cyan]" + "[/cyan] [cyan]".join(parts) + "[/cyan]")
    else:
        console.print(f"{title}: " + " ".join(parts))


def print_success_summary(
    console: Console,
    output_path: Path,
    completed: int,
    total_cells: int,
    debug: dict[str, Any] | None,
    *,
    allowed_error_count: int = 0,
    mode: OutputMode = "modern",
) -> None:
    if mode == "jsonl":
        return
    suffix = f" with {allowed_error_count} allowed error(s)" if allowed_error_count else ""
    message = (
        f"Completed {completed}/{total_cells} executable cell(s){suffix}.\n"
        f"Wrote {output_path}."
    )
    if mode == "modern":
        console.print(Panel(message, title="[green]Run complete[/green]", border_style="green"))
    else:
        console.print(message.replace("\n", " "))
    print_debug_info(console, "Modal run", debug, mode=mode)


def print_failure_summary(
    console: Console,
    output_path: Path | None,
    failure: Event,
    debug: dict[str, Any] | None,
    *,
    mode: OutputMode = "modern",
) -> None:
    if mode == "jsonl":
        return
    notebook_cell = failure.get("notebook_cell")
    cell = failure.get("cell")
    total = failure.get("total_cells")
    original = ""
    if notebook_cell and notebook_cell != cell:
        original = f" (notebook cell {notebook_cell})"
    lines = [
        (
            f"Failed at executable cell {cell}/{total}{original}, "
            f"type={failure.get('cell_type', 'unknown')}."
        )
    ]
    preview = failure.get("source_preview")
    if preview:
        lines.append(f"Source: {preview}")
    lines.append(f"{failure.get('error_type')}: {failure.get('message')}")
    if output_path is not None:
        lines.append(f"Wrote partial notebook to {output_path}.")
    body = "\n".join(lines)
    if mode == "modern":
        console.print(
            Panel(
                body,
                title="[red]Execution failed[/red]",
                border_style="red",
            )
        )
    else:
        for line in lines:
            console.print(line)
    print_debug_info(console, "Modal run", debug, mode=mode)


def print_startup_failure_summary(
    console: Console,
    output_path: Path | None,
    failure: Event,
    debug: dict[str, Any] | None,
    *,
    mode: OutputMode = "modern",
) -> None:
    if mode == "jsonl":
        return
    lines = [
        (
            "Notebook startup failed before cell execution: "
            f"{failure.get('error_type')}: {failure.get('message')}"
        )
    ]
    if output_path is not None:
        lines.append(f"Wrote partial notebook to {output_path}.")
    if mode == "modern":
        console.print(
            Panel(
                "\n".join(lines),
                title="[red]Startup failed[/red]",
                border_style="red",
            )
        )
    else:
        for line in lines:
            console.print(line)
    print_debug_info(console, "Modal run", debug, mode=mode)


def print_key_value_panel(
    console: Console,
    title: str,
    rows: list[tuple[str, str]],
    *,
    border_style: str = "cyan",
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in rows:
        table.add_row(key, value)
    console.print(Panel(table, title=title, border_style=border_style))


def print_requirements_table(console: Console, rows: list[tuple[str, str]]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim", no_wrap=True)
    table.add_column("value")
    for key, value in rows:
        table.add_row(key, value)
    console.print(Panel(table, title="[cyan]Execution plan[/cyan]", border_style="cyan"))


def print_cell_output_panel(
    console: Console,
    *,
    cell_label: str,
    stream: str | None,
    text: str,
) -> None:
    label = f"{cell_label} output"
    if stream:
        label += f" ({stream})"
    console.print(
        Panel(
            text,
            title=f"[dim]{label}[/dim]",
            border_style="bright_black",
            expand=False,
        )
    )
