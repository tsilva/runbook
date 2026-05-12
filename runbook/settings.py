from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from runbook.files import atomic_write_text
from runbook.requirements_plan import DEFAULT_OPENROUTER_MODEL


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str | None = None
    model: str = DEFAULT_OPENROUTER_MODEL


def runbook_config_dir() -> Path:
    configured = os.environ.get("RUNBOOK_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".config" / "runbook"


def runbook_env_path() -> Path:
    return runbook_config_dir() / ".env"


def init_settings_dir() -> Path:
    config_dir = runbook_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def load_openrouter_settings() -> OpenRouterSettings:
    values = _read_env_file(runbook_env_path())
    api_key = os.environ.get("OPENROUTER_API_KEY") or values.get("OPENROUTER_API_KEY")
    model = (
        os.environ.get("RUNBOOK_OPENROUTER_MODEL")
        or values.get("RUNBOOK_OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    )
    return OpenRouterSettings(api_key=api_key or None, model=model)


def save_openrouter_settings(settings: OpenRouterSettings) -> None:
    init_settings_dir()
    values = {
        "OPENROUTER_API_KEY": settings.api_key or "",
        "RUNBOOK_OPENROUTER_MODEL": settings.model or DEFAULT_OPENROUTER_MODEL,
    }
    path = runbook_env_path()
    atomic_write_text(
        path,
        "\n".join(f"{key}={_escape_env_value(value)}" for key, value in values.items())
        + "\n",
        mode=0o600,
    )


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _unescape_env_value(value.strip())
    return values


def _escape_env_value(value: str) -> str:
    if not value:
        return ""
    if any(char.isspace() for char in value) or any(char in value for char in "\"'\\#"):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def _unescape_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value
