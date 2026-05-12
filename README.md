# Runbook

Runbook executes a notebook remotely on Modal and writes an executed `.ipynb`
locally. It supports `.ipynb` files and Jupytext `.py` percent notebooks.

## Install

```bash
uv tool install . --editable
modal setup
```

Runbook exposes the console script declared in `pyproject.toml`:

```bash
runbook --help
```

## Usage

```bash
runbook input.ipynb --gpu A10 --timeout 7200 --output runs/output.ipynb
```

CPU execution is the default when `--gpu` is omitted:

```bash
runbook analysis.py --output runs/analysis.ipynb
```

Useful options:

```bash
runbook input.ipynb \
  --timeout 3600 \
  --cpu 4 \
  --memory 16384 \
  --secret huggingface-token \
  --volume model-cache:/models \
  --kernel-name python3 \
  --allow-errors
```

## Behavior

- Execution always happens inside one Modal container.
- Runbook starts one Jupyter kernel and executes code cells one by one so state
  persists between cells.
- Progress is based on completed code cells.
- Cell outputs, rich display outputs, stdout/stderr, and tracebacks are
  preserved in the written output notebook.
- By default, execution stops at the first cell error, writes the partial
  notebook, and exits nonzero.
- `--allow-errors` continues after cell errors and writes a completed notebook
  containing those errors.

## Current Limitations

- Live intra-cell stdout streaming is not implemented in v1; stdout/stderr are
  captured in the output notebook.
- The local notebook file is uploaded as notebook JSON. Additional local data
  files are not automatically uploaded; use Modal Volumes or bake data into the
  selected image.
- `--image` accepts a public registry image and adds Python 3.11 plus Runbook's
  notebook execution dependencies.
- The CLI prints the best Modal debug identifiers available from the SDK. Exact
  logs URLs are printed only if exposed by the runtime.
