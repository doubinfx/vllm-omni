"""Keep CANN's mixed BF16/F32 backward and allocate dP in probability dtype.

torch_npu 2.7 allocates both gradient outputs with dY's BF16 dtype. The mixed
CANN kernel needs FP32 dP for FP32 probabilities. Forward and CANN arithmetic
are unchanged; this adapter supplies the correct output descriptor/storage.
HY's existing explicit FP32 backward is safe; this avoids its tensor casts.
"""

import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

import torch


_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths

    source = Path(__file__).with_name("csrc") / "mixed_unpermute.cpp"
    npu = Path(torch_npu.__file__).parent
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")).resolve()
    version = torch.__version__ + torch_npu.__version__ + str(torch._C._GLIBCXX_USE_CXX11_ABI) + str(cann)
    key = hashlib.sha256(source.read_bytes() + version.encode()).hexdigest()[:20]
    cache = os.environ.get("VEOMNI_NPU_OPS_CACHE", str(Path(__file__).resolve().parents[3] / ".local/npu_ops"))
    directory = Path(cache) / key
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "mixed_unpermute.so"
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix(".tmp.so")
            includes = [*include_paths(), str(npu / "include"), str(cann / "include")]
            paths = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
            command = [
                os.environ.get("CXX", "c++"), "-shared", "-fPIC", "-O2", "-std=c++17",
                "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
                str(source), "-o", str(temporary),
            ]
            command += ["-I" + path for path in includes]
            command += ["-L" + path for path in paths]
            command += ["-ltorch_cpu", "-ltorch", "-lc10", "-ltorch_npu", "-lascendcl", "-lopapi"]
            command += ["-Wl,-rpath," + path for path in paths]
            subprocess.run(command, check=True)
            temporary.replace(library)
    torch.ops.load_library(str(library))
    _loaded = True


def backward(x, dy, inverse, probabilities):
    load()
    return torch.ops.veomni_mixed_unpermute.backward(x, dy, inverse, probabilities)


def install():
    """Install without loading/building an NPU library on CPU-only imports."""
    import torch_npu

    native = torch_npu.npu_moe_token_unpermute
    if getattr(native, "_hy_native_mixed_unpermute", False):
        return

    class Unpermute(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, inverse, probabilities):
            ctx.save_for_backward(x, inverse, probabilities)
            return native(x, inverse, probs=probabilities)

        @staticmethod
        def backward(ctx, dy):
            x, inverse, probabilities = ctx.saved_tensors
            dx, dp = backward(x, dy.contiguous(), inverse, probabilities)
            return dx, None, dp

    def apply(x, inverse, probs=None, **kwargs):
        if (
            torch.is_grad_enabled()
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and probs is not None
            and probs.dtype == torch.float32
            and not kwargs
        ):
            return Unpermute.apply(x.contiguous(), inverse.contiguous(), probs.contiguous())
        return native(x, inverse, probs=probs, **kwargs)

    # Same contract as the existing HY106 adapter; avoid stacking either hook.
    apply._hy_native_mixed_unpermute = True
    apply._hy_fp32_probability_backward = True
    apply._veomni_fp32_probability_backward = True
    torch_npu.npu_moe_token_unpermute = apply
