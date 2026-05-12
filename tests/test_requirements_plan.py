from __future__ import annotations

import nbformat
import yaml

from runbook.requirements_plan import (
    NotebookRequirements,
    companion_requirements_path,
    load_or_generate_requirements,
    parse_requirements,
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
                "model": "test-model",
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
    saved = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    assert saved["packages"]["pip"] == ["torch"]
    assert saved["planner"]["generated_at"]


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
