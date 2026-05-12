<div align="center">
  <img src="./logo.png" alt="Runbook logo" width="520" />

  **📓 Execute notebooks on remote Modal compute 📓**
</div>

Runbook is a Python CLI that runs a local Jupyter notebook remotely on Modal and
writes the executed `.ipynb` back to disk. It supports native `.ipynb` files and
Jupytext `.py` percent notebooks.

Use it when a notebook needs Modal resources such as GPUs, larger CPU or memory
requests, secrets, volumes, or a custom registry image.

## Install

```bash
git clone https://github.com/tsilva/runbook.git
cd runbook
uv tool install . --editable
modal setup
```

Run the CLI from any shell:

```bash
runbook --help
```

## Commands

```bash
runbook input.ipynb --gpu A10 --timeout 7200 --output runs/output.ipynb
runbook analysis.py --output runs/analysis.ipynb
runbook input.ipynb --cpu 4 --memory 16384 --secret huggingface-token
runbook input.ipynb --volume model-cache:/models --kernel-name python3
runbook input.ipynb --allow-errors
python3 -m pytest  # run tests from a dev environment with pytest installed
```

CPU execution is the default when `--gpu` is omitted.

## Notes

- Execution happens inside one Modal container.
- Runbook starts one Jupyter kernel and executes code cells one by one, so state
  persists between cells.
- Cell outputs, rich display outputs, stdout/stderr, and tracebacks are
  preserved in the written notebook.
- By default, execution stops at the first cell error, writes the partial
  notebook, and exits nonzero.
- `--allow-errors` continues after cell errors and writes a completed notebook
  containing those errors.
- The local notebook is uploaded as notebook JSON. Additional local data files
  are not uploaded; use Modal Volumes or bake data into the selected image.
- `--image` accepts a public registry image and adds Python 3.11 plus Runbook's
  notebook execution dependencies.
- Live intra-cell stdout streaming is not implemented; stdout/stderr are
  captured in the output notebook.

## Architecture

![Runbook architecture diagram](./architecture.png)

## License

No license file is currently present.
