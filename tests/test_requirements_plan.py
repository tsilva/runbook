from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import nbformat
import yaml

from runbook.requirements_plan import (
    ModalRequirements,
    NotebookRequirements,
    PackageRequirements,
    PlannerMetadata,
    RequirementsConfigError,
    RuntimeRequirements,
    _call_openrouter,
    _parse_openrouter_completion,
    companion_requirements_path,
    load_or_generate_requirements,
    notebook_source_hash,
    parse_requirements,
    requirements_diff_lines,
    requirements_summary_lines,
    requirements_to_dict,
)


def _notebook_json(source: str = "import pandas as pd"):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    return nbformat.writes(notebook)


def test_load_or_generate_uses_existing_companion_without_llm(monkeypatch, tmp_path):
    input_path = tmp_path / "analysis.ipynb"
    input_path.write_text(_notebook_json(), encoding="utf-8")
    requirements_path = companion_requirements_path(input_path)
    requirements_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "runtime": {
                    "image": "python:3.11",
                    "gpu": None,
                    "cpu": 2,
                    "memory": 4096,
                    "timeout": 900,
                    "kernel_name": "python3",
                },
                "packages": {"pip": ["pandas"], "apt": []},
                "modal": {"secrets": [], "volumes": []},
                "planner": {
                    "provider": "openrouter",
                    "model": "test-model",
                    "generated_at": "2026-05-12T00:00:00+00:00",
                    "confidence": 0.9,
                    "notes": ["Uses pandas."],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fail_generate(*args, **kwargs):
        raise AssertionError("planner should not be called")

    monkeypatch.setattr("runbook.requirements_plan.generate_requirements", fail_generate)

    result = load_or_generate_requirements(input_path, _notebook_json())

    assert result.generated is False
    assert result.path == requirements_path
    assert result.requirements.runtime.image == "python:3.11"
    assert result.requirements.packages.pip == ["pandas"]


def test_load_or_generate_calls_openrouter_and_writes_yaml(monkeypatch, tmp_path):
    input_path = tmp_path / "analysis.ipynb"
    notebook_json = _notebook_json("import torch\ntorch.cuda.is_available()")
    input_path.write_text(notebook_json, encoding="utf-8")

    def fake_openrouter(notebook_text, model, *, api_key=None):
        assert "torch.cuda.is_available" in notebook_text
        assert model == "test-model"
        assert api_key is None
        return {
            "version": 1,
            "runtime": {
                "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
                "gpu": "T4",
                "cpu": None,
                "memory": None,
                "timeout": 1200,
                "kernel_name": "python3",
            },
            "packages": {"pip": ["torch"], "apt": []},
            "modal": {"secrets": [], "volumes": []},
            "planner": {
                "provider": "openrouter",
                "model": "wrong-model",
                "generated_at": None,
                "confidence": 0.8,
                "notes": ["Notebook references CUDA."],
            },
        }

    monkeypatch.setattr("runbook.requirements_plan._call_openrouter", fake_openrouter)

    result = load_or_generate_requirements(input_path, notebook_json, model="test-model")

    assert result.generated is True
    assert result.path.exists()
    assert result.requirements.runtime.gpu == "T4"
    assert result.requirements.planner.model == "test-model"
    assert result.requirements.planner.source_hash == notebook_source_hash(notebook_json)
    saved = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    assert saved["packages"]["pip"] == ["torch"]
    assert saved["planner"]["model"] == "test-model"
    assert saved["planner"]["generated_at"]
    assert saved["planner"]["source_hash"] == notebook_source_hash(notebook_json)


def test_requirements_parse_dedupes_packages():
    requirements = parse_requirements(
        {
            "version": 1,
            "packages": {"pip": ["pandas", "pandas"], "apt": ["ffmpeg", "ffmpeg"]},
        },
        source="test",
    )

    assert requirements.packages.pip == ["pandas"]
    assert requirements.packages.apt == ["ffmpeg"]
    assert requirements_to_dict(NotebookRequirements())["runtime"]["timeout"] == 3600


def test_requirements_parse_forces_build_toolchain_for_triton():
    requirements = parse_requirements(
        {
            "version": 1,
            "runtime": {"build_toolchain": False},
            "packages": {"pip": ["triton==3.4.0", "unsloth_zoo"], "apt": []},
        },
        source="test",
    )

    assert requirements.runtime.build_toolchain is True


def test_requirements_summary_and_diff_are_concise():
    previous = NotebookRequirements(runtime=RuntimeRequirements(image="python:3.11"))
    current = NotebookRequirements(
        runtime=RuntimeRequirements(image="python:3.12", gpu="T4"),
        packages=PackageRequirements(pip=["pandas"]),
        planner=PlannerMetadata(confidence=0.75, notes=["Uses pandas."]),
    )

    summary = requirements_summary_lines(current)
    diff = requirements_diff_lines(previous, current)

    assert any("confidence=0.75" in line for line in summary)
    assert any("Planner note: Uses pandas." in line for line in summary)
    assert "Changed: runtime.image" in diff
    assert "Changed: runtime.gpu" in diff
    assert "Changed: packages.pip" in diff


def test_existing_requirements_with_mismatched_source_hash_fail(tmp_path):
    input_path = tmp_path / "analysis.ipynb"
    original_notebook = _notebook_json("import pandas as pd")
    changed_notebook = _notebook_json("import polars as pl")
    input_path.write_text(changed_notebook, encoding="utf-8")
    requirements_path = companion_requirements_path(input_path)
    requirements_path.write_text(
        yaml.safe_dump(
            requirements_to_dict(
                NotebookRequirements(
                    runtime=RuntimeRequirements(image="python:3.11"),
                    modal=ModalRequirements(),
                    planner=PlannerMetadata(
                        source_hash=notebook_source_hash(original_notebook)
                    ),
                )
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    try:
        load_or_generate_requirements(input_path, changed_notebook)
    except RequirementsConfigError as exc:
        assert "--regenerate-requirements" in str(exc)
    else:
        raise AssertionError("expected stale requirements to fail")


def test_regenerate_requirements_overwrites_stale_yaml(monkeypatch, tmp_path):
    input_path = tmp_path / "analysis.ipynb"
    original_notebook = _notebook_json("import pandas as pd")
    changed_notebook = _notebook_json("import polars as pl")
    input_path.write_text(changed_notebook, encoding="utf-8")
    requirements_path = companion_requirements_path(input_path)
    requirements_path.write_text(
        yaml.safe_dump(
            requirements_to_dict(
                NotebookRequirements(
                    planner=PlannerMetadata(
                        source_hash=notebook_source_hash(original_notebook)
                    )
                )
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fake_openrouter(notebook_text, model, api_key):
        assert "polars" in notebook_text
        return requirements_to_dict(NotebookRequirements())

    result = load_or_generate_requirements(
        input_path,
        changed_notebook,
        model="test-model",
        regenerate=True,
        planner=fake_openrouter,
    )

    assert result.generated is True
    assert result.requirements.planner.source_hash == notebook_source_hash(changed_notebook)


def test_openrouter_request_uses_high_reasoning(monkeypatch):
    captured_payload = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(requirements_to_dict(NotebookRequirements()))
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 120
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    _call_openrouter("print('hello')", "openai/gpt-5.5", api_key="sk-test")

    assert captured_payload["model"] == "openai/gpt-5.5"
    assert captured_payload["max_completion_tokens"] == 16000
    assert captured_payload["reasoning"] == {"effort": "high", "exclude": True}


def test_openrouter_retries_transient_errors(monkeypatch):
    calls = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(requirements_to_dict(NotebookRequirements()))
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                hdrs=Message(),
                fp=io.BytesIO(b"retry later"),
            )
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    _call_openrouter("print('hello')", "openai/gpt-5.5", api_key="sk-test")

    assert calls == 2


def test_openrouter_completion_accepts_parsed_payload():
    requirements = requirements_to_dict(NotebookRequirements())

    parsed = _parse_openrouter_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "parsed": requirements,
                    }
                }
            ]
        }
    )

    assert parsed == requirements


def test_openrouter_completion_explains_empty_content():
    try:
        _parse_openrouter_completion(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": None,
                        },
                    }
                ]
            }
        )
    except TypeError as exc:
        assert "finish_reason='length'" in str(exc)
    else:
        raise AssertionError("expected empty OpenRouter content to fail clearly")
