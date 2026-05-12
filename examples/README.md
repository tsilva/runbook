# Runbook Smoke Notebooks

These notebooks are small Modal smoke tests for Runbook.

## PyTorch T4

Run the native notebook:

```bash
runbook examples/pytorch_t4_smoke.ipynb \
  --gpu T4 \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --timeout 1200 \
  --output runs/pytorch_t4_smoke.executed.ipynb
```

Run the Jupytext percent notebook:

```bash
runbook examples/pytorch_t4_jupytext_smoke.py \
  --gpu T4 \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --timeout 1200 \
  --output runs/pytorch_t4_jupytext_smoke.executed.ipynb
```

Both notebooks assert that CUDA is available, the GPU name includes `T4`, and
PyTorch tensor, matrix multiplication, CPU/GPU comparison, and autograd
operations succeed.
