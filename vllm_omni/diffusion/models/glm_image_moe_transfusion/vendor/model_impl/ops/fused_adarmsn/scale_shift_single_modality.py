import torch
import triton
import triton.language as tl


@triton.jit
def _adarmsn_scale_shift_single_modality_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    weight,  # pointer to the adaptive scale weights
    bias,  # pointer to the adaptive shift biases
    Rstd,  # pointer to the 1/rms
    G,  # number of groups
    cumsum_M,  # pointer to the cumsum of M
    group_indices,
    stride: tl.constexpr,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    INDEXED: tl.constexpr = False,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)

    if INDEXED:
        gid = tl.load(group_indices + row)
    else:
        gid = 0
        for i in tl.range(0, G):
            cumsum_val = tl.load(cumsum_M + i)
            if row >= cumsum_val:
                gid = i + 1

    weight += gid * N
    bias += gid * N

    Y += row * stride
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Compute RMSNorm rstd = 1 / sqrt(mean(x^2) + eps).
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply adaptive scale/shift: rms_norm(x) * weight + bias.
    w = tl.load(weight + cols, mask=mask).to(tl.float32)
    b = tl.load(bias + cols, mask=mask).to(tl.float32)
    y = x * rstd * w + b
    tl.store(Y + cols, y, mask=mask)  # store automatically casts precision


@triton.jit
def _adarmsn_scale_shift_single_modality_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    X,  # pointer to the input
    W,  # pointer to the adaptive scale weights
    Rstd,  # pointer to the 1/rms
    Lock,  # pointer to the lock
    G,  # number of groups, i.e. data num
    cumsum_M,
    stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    gid = 0
    for i in tl.range(0, G):
        cumsum_val = tl.load(cumsum_M + i)
        if row >= cumsum_val:
            gid = i + 1
    W += gid * N
    DW += gid * GROUP_SIZE_M * N
    DB += gid * GROUP_SIZE_M * N
    Lock += gid * (2 * GROUP_SIZE_M)

    # Offset locks and weights/biases gradient pointer for parallel reduction.
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # Load data to SRAM.
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)

    # RMSNorm backward for y = (x * rstd) * w + b:
    # dx = rstd * (wdy - xhat * mean(xhat * wdy)), xhat = x * rstd.
    xhat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * c1) * rstd
    tl.store(DX + cols, dx, mask=mask)

    # Accumulate partial sums for dw/db.
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = dy.to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate.
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask).to(tl.float32)
        partial_db += tl.load(DB, mask=mask).to(tl.float32)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # Need a barrier to ensure all threads finished before releasing the lock.
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _adarmsn_scale_shift_single_modality_bwd_dwdb(
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N: tl.constexpr,  # number of columns
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    gid = tl.program_id(1)
    DW += gid * GROUP_SIZE_M * N
    DB += gid * GROUP_SIZE_M * N
    FINAL_DW += gid * N
    FINAL_DB += gid * N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0).to(tl.float32)
        db += tl.load(DB + offs, mask=mask, other=0.0).to(tl.float32)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


@triton.jit
def _adarmsn_scale_shift_indexed_bwd(
    DX, DY, DW_ROWS, DB_ROWS, X, W, Rstd, group_indices,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # No spinlocks: native index_add below reduces these per-token gradients.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = row * N + cols
    gid = tl.load(group_indices + row)
    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + gid * N + cols, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)
    xhat = x * rstd
    wdy = w * dy
    mean = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean) * rstd
    tl.store(DX + offsets, dx, mask=mask)
    tl.store(DW_ROWS + offsets, dy * xhat, mask=mask)
    tl.store(DB_ROWS + offsets, dy, mask=mask)


def adarmsn_scale_shift_single_modality_fused(x, weight, bias, cumsum_M, ctx, group_indices=None):
    y = torch.empty_like(x)
    # The indexed branch never reads G/cumsum_M. A fixed placeholder avoids
    # generating a new Triton specialization when the timestep count changes.
    G = 0 if group_indices is not None else cumsum_M.shape[0]
    sum_M, N = x.shape
    rstd = torch.empty((sum_M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This RMSNorm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _adarmsn_scale_shift_single_modality_fwd_fused[(sum_M,)](
        x,
        y,
        weight,
        bias,
        rstd,
        G,
        None if group_indices is not None else cumsum_M,
        group_indices,
        x.stride(0),
        N,
        ctx.eps,
        BLOCK_SIZE=BLOCK_SIZE,
        INDEXED=group_indices is not None,
        num_warps=num_warps,
        num_ctas=1,
    )
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    return y, rstd


def adarmsn_scale_shift_single_modality_backward_fused(grad_output, x, weight, bias, cumsum_M, rstd, ctx):
    G = cumsum_M.shape[0]
    sum_M, N = grad_output.shape
    GROUP_SIZE_M = 64
    if N <= 8192:
        GROUP_SIZE_M = 96
    if N <= 4096:
        GROUP_SIZE_M = 128
    if N <= 1024:
        GROUP_SIZE_M = 256

    locks = torch.zeros(G, 2 * GROUP_SIZE_M, dtype=torch.int32, device=x.device)
    _dw = torch.zeros((G, GROUP_SIZE_M, N), dtype=weight.dtype, device=x.device)
    _db = torch.zeros((G, GROUP_SIZE_M, N), dtype=bias.dtype, device=x.device)
    dw = torch.empty((G, N), dtype=weight.dtype, device=x.device)
    db = torch.empty((G, N), dtype=bias.dtype, device=x.device)
    dx = torch.empty_like(grad_output)

    _adarmsn_scale_shift_single_modality_bwd_dx_fused[(sum_M,)](  #
        dx,
        grad_output,
        _dw,
        _db,
        x,
        weight,
        rstd,
        locks,  #
        G,
        cumsum_M,
        x.stride(0),
        N,  #
        BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        num_warps=ctx.num_warps,
    )
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]), G)  # noqa: E731
    # accumulate partial sums in separate kernel
    _adarmsn_scale_shift_single_modality_bwd_dwdb[grid](
        _dw,
        _db,
        dw,
        db,
        min(GROUP_SIZE_M, sum_M),
        N,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        BLOCK_SIZE_M=32,  #
        BLOCK_SIZE_N=128,
        num_ctas=1,
    )
    return dx, None, dw, db, None


# Data is in [vision1, vision2 ...] form.
# weight and bias group order is [vision1, vision2 ... visionN].


class FusedAdaRMSNScaleShiftSingleModality(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cumsum_M, eps=1e-5):
        ctx.eps = eps
        assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() and cumsum_M.is_contiguous(), (
            "Not implemented: Noncontiguous input."
        )
        y, rstd = adarmsn_scale_shift_single_modality_fused(x, weight, bias, cumsum_M, ctx)
        ctx.save_for_backward(x, weight, bias, cumsum_M, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, cumsum_M, rstd = ctx.saved_tensors
        assert (
            grad_output.is_contiguous()
            and x.is_contiguous()
            and weight.is_contiguous()
            and bias.is_contiguous()
            and cumsum_M.is_contiguous()
            and rstd.is_contiguous()
        ), "Not implemented: Noncontiguous input."
        grad_input, _, grad_weight, grad_bias, _ = adarmsn_scale_shift_single_modality_backward_fused(
            grad_output, x, weight, bias, cumsum_M, rstd, ctx
        )
        return grad_input, grad_weight, grad_bias, None, None


class FusedAdaRMSNScaleShiftIndexed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, group_indices, eps):
        ctx.eps = eps
        y, rstd = adarmsn_scale_shift_single_modality_fused(
            x, weight, bias, group_indices, ctx, group_indices=group_indices
        )
        ctx.save_for_backward(x, weight, bias, group_indices, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, group_indices, rstd = ctx.saved_tensors
        dy = grad_output.contiguous()
        dx = torch.empty_like(x)
        dw_rows = torch.empty_like(x, dtype=weight.dtype)
        db_rows = torch.empty_like(x, dtype=bias.dtype)
        _adarmsn_scale_shift_indexed_bwd[(x.shape[0],)](
            dx, dy, dw_rows, db_rows, x, weight, rstd, group_indices,
            N=x.shape[1], BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
        )
        dw = torch.zeros_like(weight).index_add_(0, group_indices, dw_rows)
        db = torch.zeros_like(bias).index_add_(0, group_indices, db_rows)
        return dx, dw, db, None, None


def fused_adarmsn_scale_shift_single_modality(x, weight, bias, cumsum_m, eps=1e-5, *, group_indices=None):
    """Apply fused adaptive RMSNorm scale-shift for single-modality packed sequences."""
    if group_indices is not None:
        # MoT/SelfFlow timestep groups can be interleaved. Index the compact
        # coefficient table directly instead of expanding it to token count.
        if (x.ndim != 2 or weight.ndim != 2 or bias.shape != weight.shape
                or weight.shape[1] != x.shape[1] or group_indices.ndim != 1
                or group_indices.numel() != x.shape[0]
                or group_indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Invalid indexed AdaRMSNorm shapes or group index dtype")
        if not all(t.is_contiguous() and t.device == x.device for t in (x, weight, bias, group_indices)):
            raise ValueError("Indexed AdaRMSNorm requires contiguous tensors on the same device")
        return FusedAdaRMSNScaleShiftIndexed.apply(x, weight, bias, group_indices, eps)
    return FusedAdaRMSNScaleShiftSingleModality.apply(x, weight, bias, cumsum_m, eps)
