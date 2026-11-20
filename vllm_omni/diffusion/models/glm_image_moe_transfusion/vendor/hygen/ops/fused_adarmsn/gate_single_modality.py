"""FP32 AdaRMSNorm gate, with optional residual and packed/indexed groups.

The gate itself has no normalization: y = gate * branch (+ residual).
Keep FP32 multiplication/addition until the final output store. Unlike the
legacy AdaLN gate, return exactly G coefficient gradients and use native
reduction instead of spinlocks for shared timestep groups.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_forward(
    X, W, Residual, Y, Groups,
    N: tl.constexpr, W_STRIDE: tl.constexpr, MODE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = row * N + col
    active = col < N
    if MODE == 0:
        group = row
    else:
        group = tl.load(Groups + row)
    x = tl.load(X + offset, mask=active, other=0).to(tl.float32)
    w = tl.load(W + group * W_STRIDE + col, mask=active, other=0).to(tl.float32)
    output = x * w
    if HAS_RESIDUAL:
        residual = tl.load(Residual + offset, mask=active, other=0).to(tl.float32)
        output = output + residual
    tl.store(Y + offset, output, mask=active)


@triton.jit
def _gate_backward(
    X, W, DY, DX, DW_ROWS, Groups,
    N: tl.constexpr, W_STRIDE: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = row * N + col
    active = col < N
    if MODE == 0:
        group = row
    else:
        group = tl.load(Groups + row)
    x = tl.load(X + offset, mask=active, other=0).to(tl.float32)
    w = tl.load(W + group * W_STRIDE + col, mask=active, other=0).to(tl.float32)
    dy = tl.load(DY + offset, mask=active, other=0).to(tl.float32)
    tl.store(DX + offset, dy * w, mask=active)
    tl.store(DW_ROWS + offset, dy * x, mask=active)


class FusedAdaRMSNGateSingleModality(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, groups, residual, mode):
        ctx.save_for_backward(x, weight, groups)
        ctx.mode, ctx.has_residual = mode, residual is not None
        ctx.block = triton.next_power_of_2(x.shape[1])
        ctx.num_warps = min(max(ctx.block // 256, 1), 8)
        y = torch.empty_like(x)
        if x.numel():
            _gate_forward[(x.shape[0],)](
                x, weight, residual, y, groups,
                N=x.shape[1], W_STRIDE=weight.stride(0), MODE=mode,
                HAS_RESIDUAL=ctx.has_residual, BLOCK=ctx.block,
                num_warps=ctx.num_warps, enable_fp_fusion=False,
            )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, groups = ctx.saved_tensors
        dy = grad_output.contiguous()
        dx = torch.empty_like(x)
        dw_rows = torch.empty_like(x, dtype=weight.dtype)
        if x.numel():
            _gate_backward[(x.shape[0],)](
                x, weight, dy, dx, dw_rows, groups,
                N=x.shape[1], W_STRIDE=weight.stride(0), MODE=ctx.mode, BLOCK=ctx.block,
                num_warps=ctx.num_warps, enable_fp_fusion=False,
            )
        if ctx.mode == 0:
            dw = dw_rows
        else:
            dw = torch.zeros_like(weight).index_add_(0, groups, dw_rows)
        return dx, dw, None, dy if ctx.has_residual else None, None


def fused_adarmsn_gate_single_modality(x, weight, cumsum_m=None, *, group_indices=None, residual=None):
    """Gate packed sequences, indexed groups, or already expanded token rows.

    x: [tokens, hidden]; weight: [groups, hidden] or [tokens, hidden].
    Supplying residual fuses its addition before BF16 output rounding.
    """
    if cumsum_m is not None and group_indices is not None:
        raise ValueError('Choose cumulative sequence lengths or group indices, not both')
    mode = 2 if cumsum_m is not None else (1 if group_indices is not None else 0)
    groups = cumsum_m if mode == 2 else group_indices
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_contiguous() or weight.stride(-1) != 1
            or weight.device != x.device):
        raise ValueError('AdaRMSNorm gate requires contiguous x and matching last-dimension-contiguous weights')
    if mode == 0 and weight.shape[0] != x.shape[0]:
        raise ValueError('Expanded gate weights must have one row per token')
    if groups is not None:
        if (groups.ndim != 1 or groups.device != x.device or not groups.is_contiguous()
                or groups.dtype not in (torch.int32, torch.int64)
                or groups.numel() != (weight.shape[0] if mode == 2 else x.shape[0])):
            raise ValueError('Invalid AdaRMSNorm gate group metadata')
    if residual is not None and (
        residual.shape != x.shape or residual.device != x.device
        or residual.dtype != x.dtype or not residual.is_contiguous()
    ):
        raise ValueError('Residual must match branch shape, device, dtype and layout')
    if mode == 2:
        # Resolve packed boundaries once, including zero-length groups. The
        # device kernel then uses the same scalar row lookup as indexed MoT.
        lengths = torch.diff(groups, prepend=groups.new_zeros(1))
        groups = torch.arange(weight.shape[0], device=x.device).repeat_interleave(
            lengths, output_size=x.shape[0]
        )
        mode = 1
    return FusedAdaRMSNGateSingleModality.apply(x, weight, groups, residual, mode)
