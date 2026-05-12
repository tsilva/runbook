"""Runbook notebook runner."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("runbook")
except PackageNotFoundError:  # pragma: no cover - only when running from an unpackaged tree
    __version__ = "0.1.0"
