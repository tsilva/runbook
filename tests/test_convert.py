from __future__ import annotations

import nbformat

from runbook.convert import default_output_path, read_notebook


def test_reads_ipynb(tmp_path):
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('hello')")]
    )
    path = tmp_path / "input.ipynb"
    nbformat.write(notebook, path)

    converted = read_notebook(path)
    parsed = nbformat.reads(converted.notebook_json, as_version=4)

    assert parsed.cells[0].source == "print('hello')"
    assert default_output_path(path).name == "input.executed.ipynb"


def test_reads_jupytext_percent(tmp_path):
    path = tmp_path / "notebook.py"
    path.write_text(
        "# %%\n"
        "x = 1\n"
        "\n"
        "# %% [markdown]\n"
        "# Title\n"
        "\n"
        "# %%\n"
        "print(x)\n",
        encoding="utf-8",
    )

    converted = read_notebook(path)
    parsed = nbformat.reads(converted.notebook_json, as_version=4)

    assert [cell.cell_type for cell in parsed.cells] == ["code", "markdown", "code"]
    assert parsed.cells[2].source == "print(x)"
