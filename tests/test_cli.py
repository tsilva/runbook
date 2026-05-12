from __future__ import annotations

import json

import nbformat
import pytest
from typer.testing import CliRunner

from runbook.cli import app
from runbook.requirements_plan import (
    ModalRequirements,
    NotebookRequirements,
    PackageRequirements,
    RequirementsLoadResult,
    RuntimeRequirements,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _runbook_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "config"))


def _write_input(path):
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('hello')")]
    )
    nbformat.write(notebook, path)
    return notebook


def _finished_path(path):
    return path.with_name(f"{path.stem}.finished.ipynb")


def _running_path(path):
    return path.with_name(f"{path.stem}.running.ipynb")


def _patch_requirements(monkeypatch, requirements=None, *, generated=False):
    requirements = requirements or NotebookRequirements()

    def fake_load_or_create(console, log, input_path, notebook_json, **kwargs):
        return RequirementsLoadResult(
            path=input_path.with_name(f"{input_path.name}.yaml"),
            requirements=requirements,
            generated=generated,
        )

    monkeypatch.setattr("runbook.cli._load_or_create_requirements", fake_load_or_create)


def test_cli_success_writes_output_and_prints_modal_debug(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "runs" / "output.ipynb"
    notebook = _write_input(input_path)
    _patch_requirements(monkeypatch)

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
    assert _finished_path(output_path).exists()
    assert "T+" in result.output
    assert "Modal setup and image preparation in" in result.output
    assert "Remote notebook execution in" in result.output
    assert "Cell 1/1 started" in result.output
    assert "Cell 1/1 finished in" in result.output
    assert "Wrote notebook to" in result.output
    assert "function_call_id=fc-123" in result.output
    assert result.output.count("function_call_id=fc-123") == 2
    assert "Completed 1/1 executable cell" in result.output
    written = nbformat.read(_finished_path(output_path), as_version=4)
    assert written.metadata["runbook"]["status"] == "finished"
    assert written.metadata["runbook"]["runtime"]["timeout"] == 3600
    assert written.metadata["runbook"]["modal"]["debug"]["function_call_id"] == "fc-123"


def test_cli_dry_run_preflights_without_remote_execution(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "runs" / "output.ipynb"
    _write_input(input_path)
    _patch_requirements(monkeypatch)

    class FakePreflightReport:
        checks = ["static plan ok"]
        warnings = ["gpu shape warning"]

    def fail_stream(notebook_json, options):
        raise AssertionError("dry run should not start remote execution")

    monkeypatch.setattr("runbook.cli.preflight_modal_run", lambda options: FakePreflightReport())
    monkeypatch.setattr("runbook.cli.stream_remote_events", fail_stream)

    result = runner.invoke(app, [str(input_path), "--output", str(output_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Preflight: static plan ok" in result.output
    assert "Preflight warning: gpu shape warning" in result.output
    assert "Dry run complete; remote execution skipped." in result.output
    assert not _finished_path(output_path).exists()


def test_cli_plain_output_avoids_rich_panels(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    _patch_requirements(monkeypatch)

    def fake_stream(notebook_json, options):
        yield {"event": "started", "total_cells": 1, "debug": {}}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(app, [str(input_path), "--output", str(output_path), "--plain"])

    assert result.exit_code == 0, result.output
    assert "T+" in result.output
    assert "Completed 1/1 executable cell" in result.output
    assert "╭" not in result.output


def test_cli_jsonl_output_is_machine_readable(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    _patch_requirements(monkeypatch)

    def fake_stream(notebook_json, options):
        yield {
            "event": "started",
            "total_cells": 1,
            "debug": {"app_name": "runbook", "function_call_id": "fc-json"},
        }
        yield {
            "event": "cell_started",
            "cell": 1,
            "notebook_cell": 1,
            "total_cells": 1,
            "source_preview": "print('hello')",
        }
        yield {
            "event": "cell_output",
            "cell": 1,
            "notebook_cell": 1,
            "total_cells": 1,
            "name": "stdout",
            "text": "hello\n",
        }
        yield {"event": "cell_finished", "cell": 1, "completed": 1, "total_cells": 1}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(app, [str(input_path), "--output", str(output_path), "--jsonl"])

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    events = [record["event"] for record in records]
    assert "run_started" in events
    assert "cell_output" in events
    assert "run_finished" in events
    assert all(line.startswith("{") for line in result.output.splitlines() if line.strip())
    assert _finished_path(output_path).exists()


def test_cli_streams_live_notebook_updates(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('old')")]
    )
    notebook.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="old\n")
    ]
    nbformat.write(notebook, input_path)
    final_notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('live')")]
    )
    final_notebook.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="live\n")
    ]
    _patch_requirements(monkeypatch)

    def read_live():
        return nbformat.read(_running_path(output_path), as_version=4)

    def fake_stream(notebook_json, options):
        yield {"event": "started", "total_cells": 1, "debug": {}}
        assert read_live().cells[0].outputs == []
        yield {
            "event": "cell_started",
            "cell": 1,
            "notebook_cell": 1,
            "total_cells": 1,
            "source_preview": "print('live')",
        }
        live = read_live()
        assert live.metadata["runbook"]["current_cell"] == 1
        assert live.cells[0].metadata["runbook"]["execution_state"] == "running"
        yield {
            "event": "cell_output",
            "cell": 1,
            "notebook_cell": 1,
            "total_cells": 1,
            "output_type": "stream",
            "name": "stdout",
            "text": "live\n",
            "output": {
                "output_type": "stream",
                "name": "stdout",
                "text": "live\n",
            },
        }
        live = read_live()
        assert live.cells[0].outputs[0].text == "live\n"
        assert live.metadata["runbook"]["current_notebook_cell"] == 1
        yield {
            "event": "cell_finished",
            "cell": 1,
            "notebook_cell": 1,
            "completed": 1,
            "total_cells": 1,
            "status": "ok",
        }
        live = read_live()
        assert live.cells[0].metadata["runbook"]["execution_state"] == "finished"
        assert live.metadata["runbook"]["current_cell"] is None
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {
            "event": "notebook",
            "format": "ipynb-json",
            "data": nbformat.writes(final_notebook),
        }

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(app, [str(input_path), "--output", str(output_path)])

    assert result.exit_code == 0, result.output
    written = nbformat.read(_finished_path(output_path), as_version=4)
    assert written.cells[0].outputs[0].text == "live\n"
    assert written.metadata["runbook"]["status"] == "finished"
    assert _finished_path(output_path).exists()
    assert not _running_path(output_path).exists()


def test_cli_cell_failure_writes_partial_and_exits_nonzero(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    _patch_requirements(monkeypatch)

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
    assert _finished_path(output_path).exists()
    assert "Failed at executable cell 1/1" in result.output
    assert "ValueError: bad" in result.output


def test_cli_startup_failure_is_distinct(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    _write_input(input_path)
    _patch_requirements(monkeypatch)

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


def test_cli_merges_requirements_with_flag_overrides(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    _patch_requirements(
        monkeypatch,
        NotebookRequirements(
            runtime=RuntimeRequirements(
                image="pytorch/pytorch:base",
                gpu="T4",
                cpu=2,
                memory=8192,
                timeout=1200,
                kernel_name="python3",
            ),
            packages=PackageRequirements(pip=["torch", "pandas"], apt=["ffmpeg"]),
            modal=ModalRequirements(secrets=["hf-token"], volumes=["cache:/cache"]),
        ),
    )

    def fake_stream(notebook_json, options):
        assert options.image == "pytorch/pytorch:base"
        assert options.gpu == "A10"
        assert options.cpu == 2
        assert options.memory == 8192
        assert options.timeout == 7200
        assert options.kernel_name == "python3"
        assert options.pip_packages == ["torch", "pandas", "scipy"]
        assert options.apt_packages == ["ffmpeg", "git"]
        assert options.secrets == ["hf-token"]
        assert options.volumes == ["cache:/cache"]
        yield {"event": "started", "total_cells": 1, "debug": {}}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(
        app,
        [
            str(input_path),
            "--output",
            str(output_path),
            "--gpu",
            "A10",
            "--timeout",
            "7200",
            "--pip-package",
            "scipy",
            "--apt-package",
            "git",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _finished_path(output_path).exists()


def test_cli_without_llm_requires_image_for_manual_settings(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    _write_input(input_path)
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    result = runner.invoke(app, [str(input_path)])

    assert result.exit_code != 0
    assert "Companion requirements file does not exist" in result.output
    assert "runbook" in result.output
    assert "--generate-requirements" in result.output
    assert "--dry-run" in result.output
    assert "Runbook will not call the LLM automatically" in result.output
    assert not input_path.with_name("input.ipynb.yaml").exists()


def test_cli_without_llm_uses_manual_flag_settings(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    def fake_stream(notebook_json, options):
        assert options.image == "python:3.11"
        assert options.pip_packages == ["pandas"]
        assert options.apt_packages == ["git"]
        yield {"event": "started", "total_cells": 1, "debug": {}}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(
        app,
        [
            str(input_path),
            "--output",
            str(output_path),
            "--image",
            "python:3.11",
            "--pip-package",
            "pandas",
            "--apt-package",
            "git",
        ],
        input="\n",
    )

    assert result.exit_code == 0, result.output
    assert "Using CLI-provided execution requirements" in result.output
    assert _finished_path(output_path).exists()
    assert not input_path.with_name("input.ipynb.yaml").exists()


def test_cli_prompts_for_openrouter_settings_and_writes_yaml(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    def fake_openrouter(notebook_text, model, *, api_key=None):
        assert api_key == "sk-test"
        assert model == "openai/gpt-5.5"
        return {
            "version": 1,
            "runtime": {
                "image": "python:3.11",
                "gpu": None,
                "cpu": None,
                "memory": None,
                "timeout": 3600,
                "kernel_name": "python3",
            },
            "packages": {"pip": [], "apt": []},
            "modal": {"secrets": [], "volumes": []},
            "planner": {
                "provider": "openrouter",
                "model": "openai/gpt-5.5",
                "generated_at": None,
                "confidence": 0.7,
                "notes": [],
            },
        }

    def fake_stream(notebook_json, options):
        assert options.image == "python:3.11"
        yield {"event": "started", "total_cells": 1, "debug": {}}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.requirements_plan._call_openrouter", fake_openrouter)
    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(
        app,
        [str(input_path), "--output", str(output_path)],
        input="sk-test\n\n",
    )

    assert result.exit_code != 0
    assert "Companion requirements file does not exist" in result.output
    assert "OpenRouter API key" not in result.output
    assert not input_path.with_name("input.ipynb.yaml").exists()
    assert not _finished_path(output_path).exists()


def test_cli_generate_requirements_flag_prompts_and_writes_yaml(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    def fake_openrouter(notebook_text, model, *, api_key=None):
        assert api_key == "sk-test"
        assert model == "openai/gpt-5.5"
        return {
            "version": 1,
            "runtime": {
                "image": "python:3.11",
                "gpu": None,
                "cpu": None,
                "memory": None,
                "timeout": 3600,
                "kernel_name": "python3",
            },
            "packages": {"pip": [], "apt": []},
            "modal": {"secrets": [], "volumes": []},
            "planner": {
                "provider": "openrouter",
                "model": "openai/gpt-5.5",
                "generated_at": None,
                "confidence": 0.7,
                "notes": [],
            },
        }

    def fake_stream(notebook_json, options):
        assert options.image == "python:3.11"
        yield {"event": "started", "total_cells": 1, "debug": {}}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.requirements_plan._call_openrouter", fake_openrouter)
    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(
        app,
        [str(input_path), "--output", str(output_path), "--generate-requirements"],
        input="sk-test\n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Generating execution requirements with OpenRouter (openai/gpt-5.5)" in result.output
    assert "Wrote execution requirements" in result.output
    assert "Starting remote execution on Modal (image=python:3.11, gpu=none)" in result.output
    assert "Image setup can take a few minutes" in result.output
    assert input_path.with_name("input.ipynb.yaml").exists()
    assert (tmp_path / "config" / ".env").exists()


def test_modal_image_includes_build_toolchain():
    from runbook.modal_app import _build_image

    class FakeImage:
        def __init__(self):
            self.calls = []

        def apt_install(self, *packages):
            self.calls.append(("apt_install", packages))
            return self

        def pip_install(self, *packages, **kwargs):
            self.calls.append(("pip_install", packages, kwargs))
            return self

        def add_local_python_source(self, package):
            self.calls.append(("add_local_python_source", package))
            return self

    class FakeImageFactory:
        def __init__(self, image):
            self.image = image

        def from_registry(self, image_name):
            self.image.calls.append(("from_registry", image_name))
            return self.image

        def debian_slim(self, python_version):
            self.image.calls.append(("debian_slim", python_version))
            return self.image

    class FakeModal:
        def __init__(self, image):
            self.Image = FakeImageFactory(image)

    image = FakeImage()
    _build_image(
        FakeModal(image),
        "pytorch/pytorch:example",
        pip_packages=["torch", "nbformat"],
        apt_packages=["ffmpeg", "build-essential"],
        python_version="3.12",
        pip_index_url="https://example.test/simple",
        pip_extra_index_urls=["https://extra.example.test/simple"],
    )

    assert ("apt_install", ("build-essential", "ffmpeg")) in image.calls
    assert (
        "pip_install",
        ("nbformat", "nbclient", "ipykernel", "torch"),
        {
            "index_url": "https://example.test/simple",
            "extra_index_url": ["https://extra.example.test/simple"],
        },
    ) in image.calls


def test_cli_runtime_flags_reach_modal_options(monkeypatch, tmp_path):
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = _write_input(input_path)
    _patch_requirements(monkeypatch)

    def fake_stream(notebook_json, options):
        assert options.python_version == "3.12"
        assert options.build_toolchain is False
        assert options.pip_index_url == "https://example.test/simple"
        assert options.pip_extra_index_urls == ["https://extra.example.test/simple"]
        assert options.workdir.startswith("/tmp/runbook-")
        yield {"event": "started", "total_cells": 1, "debug": {}}
        yield {"event": "finished", "completed": 1, "total_cells": 1}
        yield {"event": "notebook", "format": "ipynb-json", "data": nbformat.writes(notebook)}

    monkeypatch.setattr("runbook.cli.stream_remote_events", fake_stream)

    result = runner.invoke(
        app,
        [
            str(input_path),
            "--output",
            str(output_path),
            "--python-version",
            "3.12",
            "--no-build-toolchain",
            "--pip-index-url",
            "https://example.test/simple",
            "--pip-extra-index-url",
            "https://extra.example.test/simple",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "workdir=/tmp/runbook-" in result.output
