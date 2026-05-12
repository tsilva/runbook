from __future__ import annotations

import nbformat
from typer.testing import CliRunner

from runbook.cli import app


runner = CliRunner()


def _write_input(path):
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('hello')")]
    )
    nbformat.write(notebook, path)
    return notebook


def test_cli_success_writes_output_and_prints_modal_debug(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "runs" / "output.ipynb"
    notebook = _write_input(input_path)

    def fake_stream(notebook_json, options):
        assert options.gpu is None
        assert options.timeout == 3600
        yield {
            "event": "started",
            "total_cells": 1,
            "debug": {
                "app_name": "runbook",
                "function_name": "runbook_remote_runner",
                "function_call_id": "fc-123",
                "input_id": "in-123",
                "dashboard_url": "https://modal.com/apps",
            },
        }
        yield {"event": "cell_started", "cell": 1, "notebook_cell": 1, "total_cells": 1}
        yield {
            "event": "cell_finished",
            "cell": 1,
            "notebook_cell": 1,
            "completed": 1,
            "total_cells": 1,
            "status": "ok",
        }
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(app, [str(input_path), "--output", str(output_path)])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "function_call_id=fc-123" in result.output
    assert result.output.count("function_call_id=fc-123") == 2
    assert "Completed 1/1 executable cell" in result.output


def test_cli_cell_failure_writes_partial_and_exits_nonzero(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)

    def fake_stream(notebook_json, options):
        yield {
            "event": "started",
            "total_cells": 1,
            "debug": {"app_name": "runbook", "function_name": "runner"},
        }
        yield {
            "event": "cell_failed",
            "cell": 1,
            "notebook_cell": 1,
            "total_cells": 1,
            "cell_type": "code",
            "error_type": "ValueError",
            "message": "bad",
            "source_preview": "raise ValueError('bad')",
            "allowed": False,
        }
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(app, [str(input_path), "--output", str(output_path)])

    assert result.exit_code != 0
    assert output_path.exists()
    assert "Failed at executable cell 1/1" in result.output
    assert "ValueError: bad" in result.output


def test_cli_startup_failure_is_distinct(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    _write_input(input_path)

    def fake_stream(notebook_json, options):
        yield {
            "event": "started",
            "total_cells": 1,
            "debug": {"app_name": "runbook", "function_name": "runner"},
        }
        yield {
            "event": "startup_failed",
            "error_type": "NoSuchKernel",
            "message": "missing",
            "traceback": "trace",
        }

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(app, [str(input_path)])

    assert result.exit_code != 0
    assert "Notebook startup failed before cell execution" in result.output
    assert "Failed at executable cell" not in result.output
