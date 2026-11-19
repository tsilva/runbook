from __future__ import annotations

import pytest

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
    monkeypatch.delenv("AGENTBRIDGE_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    config_dir = init_settings_dir()
    save_openrouter_settings(OpenRouterSettings(api_key="sk-test", model="openai/gpt-5.5"))

    assert config_dir == tmp_path / "runbook-config"
    assert runbook_env_path().exists()
    loaded = load_openrouter_settings()
    assert loaded.api_key == "sk-test"
    assert loaded.model == "openai/gpt-5.5"


def test_settings_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path / "runbook-config"))
    monkeypatch.delenv("AGENTBRIDGE_API_KEY", raising=False)
    monkeypatch.delenv("RUNBOOK_OPENROUTER_MODEL", raising=False)

    settings = load_openrouter_settings()

    assert settings.api_key == "not-needed"
    assert settings.model == DEFAULT_OPENROUTER_MODEL


def test_agent_settings_skip_local_credential_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNBOOK_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("RUNBOX_POLICY", "synthetic-session")
    monkeypatch.setenv("AGENTBRIDGE_API_KEY", "agentbridge_public_placeholder")
    monkeypatch.setenv("RUNBOOK_OPENROUTER_MODEL", "openai/fixture")

    def unexpected_read(path):
        pytest.fail("Agent session opened local credentials")

    monkeypatch.setattr("runbook.settings._read_env_file", unexpected_read)

    assert load_openrouter_settings() == OpenRouterSettings(
        api_key="agentbridge_public_placeholder", model="openai/fixture"
    )
    with pytest.raises(ValueError, match="cannot write local AgentBridge settings"):
        save_openrouter_settings(OpenRouterSettings(api_key="synthetic-key"))
    assert not runbook_env_path().exists()
