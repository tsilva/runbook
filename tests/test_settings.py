from __future__ import annotations

from runbook.requirements_plan import DEFAULT_OPENROUTER_MODEL
from runbook.settings import (
    OpenRouterSettings,
    init_settings_dir,
    load_openrouter_settings,
    runbook_env_path,
    save_openrouter_settings,
)


def test_settings_dir_and_env_file_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "runbook-config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    config_dir = init_settings_dir()
    save_openrouter_settings(
        OpenRouterSettings(api_key="sk-test", model="openai/gpt-5.5")
    )

    assert config_dir == tmp_path / "runbook-config"
    assert runbook_env_path().exists()
    loaded = load_openrouter_settings()
    assert loaded.api_key == "sk-test"
    assert loaded.model == "openai/gpt-5.5"


def test_settings_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "runbook-config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    settings = load_openrouter_settings()

    assert settings.api_key is None
    assert settings.model == DEFAULT_OPENROUTER_MODEL
