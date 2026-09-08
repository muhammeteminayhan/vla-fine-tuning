"""Phase 0 checks 1-4: driver, open kernel module, torch/CUDA, sm_120, free VRAM.

Run inside the `lerobot` conda env. Prints raw values only; no interpretation.
"""

import subprocess
import sys


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


print("=" * 70)
print("CHECK 1 - nvidia-smi")
print("=" * 70)
print(sh("nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total "
         "--format=csv"))

print()
print("=" * 70)
print("CHECK 2 - open kernel module (expect: Dual MIT/GPL)")
print("=" * 70)
print(sh("modinfo nvidia | grep -E '^(license|version)'"))

print()
print("=" * 70)
print("CHECK 3 - torch / CUDA / compute capability")
print("=" * 70)
import torch  # noqa: E402

print(f"torch.__version__            = {torch.__version__}")
print(f"torch.version.cuda           = {torch.version.cuda}")
print(f"torch.cuda.is_available()    = {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FAIL: CUDA not available")
    sys.exit(1)
print(f"torch.cuda.get_device_name() = {torch.cuda.get_device_name(0)}")
print(f"get_device_capability()      = {torch.cuda.get_device_capability(0)}")
print(f"torch.cuda.get_arch_list()   = {torch.cuda.get_arch_list()}")

cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print(f"this GPU needs               = {arch}")
print(f"{arch} in arch_list           = {arch in torch.cuda.get_arch_list()}")

# The decisive test: a real kernel launch. An unsupported arch fails HERE,
# not in the metadata above.
print("\n-- real GPU matmul (this is what actually proves sm_120 works) --")
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
c = a @ b
torch.cuda.synchronize()
print(f"fp32 matmul 4096x4096 -> {tuple(c.shape)}, sum={c.sum().item():.4f}")

ah = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
bh = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
ch = ah @ bh
torch.cuda.synchronize()
print(f"bf16 matmul 4096x4096 -> {tuple(ch.shape)}, dtype={ch.dtype}")

print()
print("=" * 70)
print("CHECK 4 - free VRAM (this is the training budget)")
print("=" * 70)
del a, b, c, ah, bh, ch
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f"mem_get_info() free  = {free:>14,} B = {free / 1024**3:.3f} GiB")
print(f"mem_get_info() total = {total:>14,} B = {total / 1024**3:.3f} GiB")
print(f"used by others       = {(total - free) / 1024**2:.1f} MiB")
