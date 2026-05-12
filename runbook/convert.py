from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbformat


@dataclass(frozen=True)
class ConvertedNotebook:
    """Notebook content ready to upload to Modal."""

    source_path: Path
    notebook_json: str


class NotebookConversionError(RuntimeError):
    """Raised when a local input cannot be converted to notebook JSON."""


def read_notebook(path: Path) -> ConvertedNotebook:
    """Read an ipynb or Jupytext percent notebook and return ipynb JSON."""

    source_path = path.expanduser().resolve()
    if not source_path.exists():
        raise NotebookConversionError(f"Input notebook does not exist: {source_path}")
    if not source_path.is_file():
        raise NotebookConversionError(f"Input path is not a file: {source_path}")

    suffix = source_path.suffix.lower()
    try:
        if suffix == ".ipynb":
            notebook = nbformat.read(source_path, as_version=4)
        elif suffix == ".py":
            import jupytext

            notebook = jupytext.read(source_path)
        else:
            raise NotebookConversionError(
                "Unsupported input format. Expected .ipynb or Jupytext .py percent file."
            )
    except NotebookConversionError:
        raise
    except Exception as exc:  # pragma: no cover - exact parser exceptions vary
        raise NotebookConversionError(f"Could not read notebook: {exc}") from exc

    try:
        notebook_json = nbformat.writes(notebook)
    except Exception as exc:  # pragma: no cover - nbformat controls this path
        raise NotebookConversionError(f"Could not serialize notebook: {exc}") from exc

    return ConvertedNotebook(source_path=source_path, notebook_json=notebook_json)


def default_output_path(input_path: Path) -> Path:
    """Return the default executed ipynb path for an input notebook."""

    return input_path.with_name(f"{input_path.stem}.executed.ipynb")
