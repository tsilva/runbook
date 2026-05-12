from __future__ import annotations

from typing import Any, Literal, TypedDict


class DebugInfo(TypedDict, total=False):
    app_name: str
    function_name: str
    function_call_id: str | None
    input_id: str | None
    dashboard_url: str
    resources: dict[str, Any]


class StartedEvent(TypedDict):
    event: Literal["started"]
    total_cells: int
    debug: DebugInfo


class CellStartedEvent(TypedDict):
    event: Literal["cell_started"]
    cell: int
    notebook_cell: int
    total_cells: int
    cell_type: str
    source_preview: str


class CellFailedEvent(TypedDict):
    event: Literal["cell_failed"]
    cell: int
    notebook_cell: int
    total_cells: int
    cell_type: str
    error_type: str
    message: str
    traceback: str
    source_preview: str
    allowed: bool


class CellFinishedEvent(TypedDict):
    event: Literal["cell_finished"]
    cell: int
    notebook_cell: int
    completed: int
    total_cells: int
    status: Literal["ok", "error"]


class CellOutputEvent(TypedDict, total=False):
    event: Literal["cell_output"]
    cell: int
    notebook_cell: int
    total_cells: int
    output_type: str
    name: str
    text: str
    output: dict[str, Any]


class StartupFailedEvent(TypedDict):
    event: Literal["startup_failed"]
    error_type: str
    message: str
    traceback: str


class FinishedEvent(TypedDict):
    event: Literal["finished"]
    completed: int
    total_cells: int


class NotebookEvent(TypedDict):
    event: Literal["notebook"]
    format: Literal["ipynb-json"]
    data: str


RunbookEvent = (
    StartedEvent
    | CellStartedEvent
    | CellFailedEvent
    | CellFinishedEvent
    | CellOutputEvent
    | StartupFailedEvent
    | FinishedEvent
    | NotebookEvent
)

Event = RunbookEvent
