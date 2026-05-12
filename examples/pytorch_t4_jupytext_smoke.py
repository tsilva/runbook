# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Runbook PyTorch T4 Smoke Test
#
# This notebook is intended to be executed by Runbook on Modal with `--gpu T4`.
# It verifies CUDA availability, confirms the GPU model, and performs real
# PyTorch tensor operations on the GPU.

# %%
import platform
import torch

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")

assert torch.cuda.is_available(), "CUDA is not available; run with `--gpu T4`."

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}")

assert "T4" in gpu_name.upper(), f"Expected a T4 GPU, got {gpu_name!r}."

# %%
torch.manual_seed(7)

a = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
b = torch.linspace(-1.0, 1.0, steps=20, device=device).reshape(4, 5)
c = a @ b
activated = torch.relu(c)

assert c.shape == (3, 5)
assert c.device.type == "cuda"
assert torch.isfinite(activated).all().item()

print("Matrix multiply result:")
print(c)
print(f"ReLU sum: {activated.sum().item():.6f}")

# %%
gpu_tensor = torch.randn(256, 256, device=device)
gpu_result = (gpu_tensor.T @ gpu_tensor).diagonal().sum()
cpu_result = (gpu_tensor.cpu().T @ gpu_tensor.cpu()).diagonal().sum()

torch.testing.assert_close(gpu_result.cpu(), cpu_result, rtol=1e-4, atol=1e-3)

print(f"GPU norm-derived result: {gpu_result.item():.6f}")
print(f"CPU comparison result: {cpu_result.item():.6f}")

# %%
x = torch.randn(128, 128, device=device, requires_grad=True)
w = torch.randn(128, 64, device=device, requires_grad=True)
y = (x @ w).pow(2).mean()
y.backward()

assert x.grad is not None
assert w.grad is not None
assert torch.isfinite(x.grad).all().item()
assert torch.isfinite(w.grad).all().item()

print(f"Autograd loss: {y.item():.6f}")
print("PyTorch T4 smoke test passed.")
