<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/runbook/main/logo.png" alt="Runbook logo" width="520" />
  <br />
  <!-- repo-tagline:start -->
  <strong>📓 Execute notebooks on remote Modal compute 📓</strong>
  <!-- repo-tagline:end -->
</p>

Runbook executes a Jupyter notebook on Modal and saves the result locally.
It accepts `.ipynb` files and Jupytext percent-format Python notebooks.
It streams outputs to `.running.ipynb` and writes `.finished.ipynb` when the
run ends.

## Quickstart

With Python 3.11 or newer and uv, install from the checkout:

```bash
uv tool install . --editable
runbook --help
```

Use `runbook-plan input.ipynb` to write a requirements plan without running
notebook code. Once Modal is configured and the companion requirements file
is ready, run:

```bash
runbook input.ipynb --output runs/output.ipynb
```

See the [notebook execution guide](https://github.com/tsilva/runbook/blob/main/docs/guide.md)
for requirements, GPU options, local outputs, and remote execution settings.

## License

[MIT License](https://github.com/tsilva/runbook/blob/main/LICENSE).
