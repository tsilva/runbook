from __future__ import annotations

import nbformat

from runbook.execute import execute_notebook_events


def _notebook_json(*cells):
    notebook = nbformat.v4.new_notebook(cells=list(cells))
    return nbformat.writes(notebook)


def _events(notebook_json: str, *, allow_errors: bool = False, kernel_name: str = "python3"):
    return list(
        execute_notebook_events(
            notebook_json,
            allow_errors=allow_errors,
            timeout=30,
            kernel_name=kernel_name,
            debug={"app_name": "test", "function_name": "runner"},
            workdir="/tmp/runbook-tests",
        )
    )


def _output_notebook(events):
    event = [item for item in events if item.get("event") == "notebook"][-1]
    return nbformat.reads(event["data"], as_version=4)


def test_successful_ipynb_execution_preserves_stdout_and_stderr():
    events = _events(
        _notebook_json(
            nbformat.v4.new_code_cell("import sys\nprint('out')\nprint('err', file=sys.stderr)")
        )
    )
    notebook = _output_notebook(events)
    streams = notebook.cells[0].outputs

    assert [event["event"] for event in events].count("cell_finished") == 1
    assert any(output.name == "stdout" and "out" in output.text for output in streams)
    assert any(output.name == "stderr" and "err" in output.text for output in streams)
    assert events[-2]["event"] == "finished"


def test_rich_display_output_is_preserved():
    events = _events(
        _notebook_json(
            nbformat.v4.new_code_cell(
                "from IPython.display import display, HTML\n"
                "display(HTML('<strong>rich</strong>'))"
            )
        )
    )
    notebook = _output_notebook(events)
    output = notebook.cells[0].outputs[0]

    assert output.output_type == "display_data"
    assert "text/html" in output.data
    assert "<strong>rich</strong>" in output.data["text/html"]


def test_code_cell_failure_stops_and_reports_executable_and_original_indices():
    events = _events(
        _notebook_json(
            nbformat.v4.new_markdown_cell("# Intro"),
            nbformat.v4.new_code_cell("x = 1"),
            nbformat.v4.new_markdown_cell("between"),
            nbformat.v4.new_code_cell("raise ValueError('bad')"),
            nbformat.v4.new_code_cell("print('after')"),
        )
    )
    failure = [event for event in events if event.get("event") == "cell_failed"][0]
    notebook = _output_notebook(events)

    assert failure["cell"] == 2
    assert failure["notebook_cell"] == 4
    assert failure["total_cells"] == 3
    assert failure["cell_type"] == "code"
    assert failure["error_type"] == "ValueError"
    assert failure["message"] == "bad"
    assert "raise ValueError" in failure["source_preview"]
    assert events[-1]["event"] == "notebook"
    assert "ValueError" in "\n".join(notebook.cells[3].outputs[-1].traceback)
    assert notebook.cells[4].outputs == []


def test_allow_errors_continues_after_error_and_preserves_traceback():
    events = _events(
        _notebook_json(
            nbformat.v4.new_code_cell("raise RuntimeError('keep going')"),
            nbformat.v4.new_code_cell("print('after')"),
        ),
        allow_errors=True,
    )
    notebook = _output_notebook(events)
    failures = [event for event in events if event.get("event") == "cell_failed"]

    assert failures[0]["allowed"] is True
    assert [event["event"] for event in events].count("cell_finished") == 2
    assert events[-2]["event"] == "finished"
    assert "RuntimeError" in "\n".join(notebook.cells[0].outputs[-1].traceback)
    assert notebook.cells[1].outputs[0].text.strip() == "after"


def test_startup_failure_is_reported_distinctly():
    events = _events(
        _notebook_json(nbformat.v4.new_code_cell("print('never')")),
        kernel_name="definitely-missing-kernel",
    )

    assert any(event.get("event") == "started" for event in events)
    startup = [event for event in events if event.get("event") == "startup_failed"][0]
    assert startup["error_type"]
    assert not any(event.get("event") == "cell_failed" for event in events)


def test_non_cell_execution_exception_is_reported_as_cell_failure(monkeypatch):
    class FakeKernelContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeNotebookClient:
        def __init__(self, *args, **kwargs):
            pass

        def setup_kernel(self, **kwargs):
            return FakeKernelContext()

        def execute_cell(self, *args, **kwargs):
            raise TimeoutError("cell timed out")

    monkeypatch.setattr("runbook.execute.NotebookClient", FakeNotebookClient)

    events = _events(_notebook_json(nbformat.v4.new_code_cell("while True: pass")))
    failure = [event for event in events if event.get("event") == "cell_failed"][0]

    assert failure["error_type"] == "TimeoutError"
    assert failure["message"] == "cell timed out"
    assert not any(event.get("event") == "startup_failed" for event in events)
