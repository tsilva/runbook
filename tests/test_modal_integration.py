from __future__ import annotations

import os
from typing import cast

import nbformat
import pytest

from runbook.events import NotebookEvent
from runbook.modal_app import ModalRunOptions, stream_remote_events


@pytest.mark.modal_integration
def test_real_modal_execution_round_trip():
    if not os.environ.get("RUNBOOK_MODAL_INTEGRATION"):
        pytest.skip("Set RUNBOOK_MODAL_INTEGRATION=1 to run against Modal.")

    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("x = 40 + 2\nprint(x)")]
    )
    events = list(
        stream_remote_events(
            nbformat.writes(notebook),
            ModalRunOptions(timeout=300, gpu=None, allow_errors=False),
        )
    )
    output_event = cast(
        NotebookEvent,
        [event for event in events if event.get("event") == "notebook"][-1],
    )
    executed = nbformat.reads(output_event["data"], as_version=4)

    assert any(event.get("event") == "started" for event in events)
    assert any(event.get("event") == "finished" for event in events)
    assert executed.cells[0].outputs[0].text.strip() == "42"
