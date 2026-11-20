from typing import Literal

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .base import get_device_type


class FlowMatchNativeResolutionScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 25,
        seed: int | None = None,
        train_weight: Literal["uniform", "logit_normal"] = "logit_normal",
        base_seqlen: int = 256,
        shift_seqlen: int = 4096,
        base_shift: float = 0.5,
        shift: float = 1.15,
        use_fix_shift: bool = False,
        logit_mean: float = 0.0,
        logit_mean_end: float = 0.0,
        logit_mean_seqlen: int = 4096,
        logit_mean_end_seqlen: int = 16384,
        logit_std: float = 1.0,
    ):
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.generator = torch.Generator(device=get_device_type()).manual_seed(seed)
        self.train_weight = train_weight
        self.base_seqlen = base_seqlen
        self.shift_seqlen = shift_seqlen
        self.base_shift = base_shift
        self.shift = shift
        self.use_fix_shift = use_fix_shift
        # 训练时间步采样的 logit-normal 位置/尺度(SD3 §5.1 的 m / s):
        #   x ~ sigmoid(N(m, s)) → 再过分辨率 timeshift
        # m>0 把质量往高噪声端(t→1)推,m<0 往低噪声端推;s 控制集中度。与 timeshift
        # **正交**:timeshift(α)是分辨率相关的,改它会连带改各分辨率之间的相对关系
        # (想把 1024 的 t>0.8 从 41% 抬到 60%,shift 得 1.15→1.64,2048 的 α 会从 25 炸到
        # 197);m 是加在 logit 上的偏置,抬哪一段、抬多少可以单独指定。
        #
        # 与 hi_band_mask(train.py 的 selfflow_hiband_frac)的分工:
        #   m    平移整条曲线(形状不变),抬高噪必然同时抬薄尾,低噪端一起被抽走;
        #   band 是加一块支架,「垫多少」直接就是概率质量,不动原曲线形状。
        # 两者可叠加:band 命中的 item 直接改写成 U(lo,hi),没命中的照常走这条曲线。
        #
        # m 本身按 seqlen 线性插值(两端 clamp,不外推):
        #   L ≤ logit_mean_seqlen      → logit_mean          (默认 4096 = 1024²)
        #   L ≥ logit_mean_end_seqlen  → logit_mean_end       (默认 16384 = 2048²)
        #   之间                        → 线性
        # 所以「抬高 1024、到 2048 恰好无影响」= logit_mean=m0, logit_mean_end=0.0。
        # 两端 clamp 是刻意的:不 clamp 的话 512²(L=1024)会被外推到比 1024 更大的 m,
        # 4096² 会拿到负 m(反而往低噪声推)—— 都不是想要的。
        # 只作用于训练采样(sample_train_timesteps),推理 t 网格
        # (get_infernece_timesteps)完全不受影响。
        # 全默认(0.0 / 0.0 / 1.0)= 历史行为(与改动前逐位一致)。
        self.logit_mean = logit_mean
        self.logit_mean_end = logit_mean_end
        self.logit_mean_seqlen = logit_mean_seqlen
        self.logit_mean_end_seqlen = logit_mean_end_seqlen
        self.logit_std = logit_std
        assert logit_mean == logit_mean_end or logit_mean_end_seqlen > logit_mean_seqlen, (
            "logit_mean 斜坡的两个锚点 seqlen 必须 end > start"
        )

    def _logit_mean_per_item(self, seqlens: torch.Tensor) -> "torch.Tensor | float":
        """每个 item 按自己的 seqlen 取 m(两端 clamp 的线性插值)。"""
        if self.logit_mean == self.logit_mean_end:
            return self.logit_mean  # 常数(含全 0 的默认情形),省掉逐项计算
        w = (seqlens.float() - self.logit_mean_seqlen) / (self.logit_mean_end_seqlen - self.logit_mean_seqlen)
        return self.logit_mean + (self.logit_mean_end - self.logit_mean) * w.clamp(0.0, 1.0)

    def get_infernece_timesteps(self, grid_thw: torch.Tensor, force_zero: torch.Tensor | None = None) -> torch.Tensor:
        bsz = grid_thw.shape[0]
        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1, device=grid_thw.device, dtype=torch.float32)
        timesteps = timesteps.unsqueeze(0).repeat_interleave(bsz, dim=0)  # [B, T]
        if self.use_fix_shift:
            shift_alpha = torch.tensor([self.shift], device=grid_thw.device, dtype=torch.float32).unsqueeze(1)
        else:
            shift_mu = self.base_shift + (self.shift - self.base_shift) * (
                (grid_thw.prod(dim=-1) - self.base_seqlen) / (self.shift_seqlen - self.base_seqlen)
            )
            shift_alpha = torch.exp(shift_mu).unsqueeze(1)
        timesteps = shift_alpha * timesteps / (1 + (shift_alpha - 1) * timesteps)
        if force_zero is not None:
            timesteps[force_zero] = 0.0
        return timesteps

    def sample_train_timesteps(
        self, grid_thw: torch.Tensor, min_t: float = 0.0,
        hi_band_mask: torch.Tensor | None = None,
        hi_band: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """独立采一份 per-item 训练时间步(采样 + timeshift + min_t clamp),[bsz]。"""
        bsz = grid_thw.shape[0]
        if self.train_weight == "uniform":
            timesteps = torch.rand(bsz, generator=self.generator, device=self.generator.device)
        elif self.train_weight == "logit_normal":
            # 先取标准正态,再做 x = sigmoid(m + s·z):m 逐 item(按 seqlen 插值)。
            # 默认 m=0 / s=1 时 sigmoid(0 + 1·z) 与原来的 sigmoid(N(0,1)) 逐位相同
            # (乘 1.0 / 加 0.0 在浮点上是精确的),所以历史训练可复现。
            z = torch.normal(mean=0.0, std=1.0, size=(bsz,), generator=self.generator, device=self.generator.device)
            m = self._logit_mean_per_item(grid_thw.prod(dim=-1))
            if isinstance(m, torch.Tensor):
                m = m.to(z.device)
            timesteps = torch.sigmoid(m + self.logit_std * z)
        else:
            raise NotImplementedError(f"{self.train_weight=} is not implemented")

        # ZWD samples the band in pre-shift coordinates. Draw a full vector
        # whenever enabled, including empty masks, to preserve the RNG stream.
        if hi_band_mask is not None:
            if hi_band is None or not 0.0 <= hi_band[0] < hi_band[1] <= 1.0:
                raise ValueError("hi_band must satisfy 0 <= lo < hi <= 1")
            u = torch.rand(bsz, generator=self.generator, device=self.generator.device)
            timesteps = torch.where(
                hi_band_mask.bool(), hi_band[0] + (hi_band[1] - hi_band[0]) * u, timesteps
            )

        # time shifting
        if self.use_fix_shift:
            # add_noise 里 timesteps 是 1D [bsz](不同于 get_infernece_timesteps 的 2D
            # [B,T])→ shift_alpha 用 0-dim 标量,与非 fix 分支 exp(shift_mu) 的 [bsz] 一样
            # 广播成 [bsz];**不能 .unsqueeze(1)**(那会把 timesteps 广播成 [1,bsz]→
            # timesteps[force_zero] 与 [bsz] mask 形状不匹配 IndexError)。
            shift_alpha = torch.tensor(self.shift, device=grid_thw.device, dtype=torch.float32)
        else:
            shift_mu = self.base_shift + (self.shift - self.base_shift) * (
                (grid_thw.prod(dim=-1) - self.base_seqlen) / (self.shift_seqlen - self.base_seqlen)
            )
            shift_alpha = torch.exp(shift_mu)
        timesteps = shift_alpha * timesteps / (1 + (shift_alpha - 1) * timesteps)
        if min_t > 0.0:
            # 训练 σ 下限(LakonLab gaussian_flow.py 的 eps=1e-4 同款):在构造 x_t 之前
            # clamp,保证 x_t 与 σ 自洽,下游 1/σ 的放大被结构性封顶(asymflow 需要)。
            # 在 force_zero 之前应用,不影响被强制置 0 的 item。
            timesteps = timesteps.clamp_min(min_t)
        return timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        grid_thw: torch.Tensor,
        force_zero: torch.Tensor,
        noise_scale: torch.Tensor | None = None,
        min_t: float = 0.0,
        force_one: torch.Tensor | None = None,
        hi_band_mask: torch.Tensor | None = None,
        hi_band: tuple[float, float] | None = None,
        noise_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            size=original_samples.shape, device=original_samples.device,
            generator=self.generator, dtype=noise_dtype,
        )
        timesteps = self.sample_train_timesteps(grid_thw, min_t=min_t, hi_band_mask=hi_band_mask, hi_band=hi_band)
        timesteps[force_zero] = 0.0
        if force_one is not None:
            timesteps[force_one] = 1.0

        # adaptive lens, noise mixing, TODO(wwx): use torch.comile or triton reduce the io cost
        timesteps_repeat = timesteps.repeat_interleave(grid_thw.prod(dim=-1)).unsqueeze(1)  # [L_total, 1]
        # pixel-space diffusion: per-item 分辨率自适应 noise 端缩放(noise_scale = sqrt(seqlen/256))。
        # None → 标准 flow(vae 模式)。注意返回的 noise 也被缩放,故下游 v-target = noise - x0 自洽。
        if noise_scale is not None:
            ns = (
                noise_scale.to(noise.device, noise.dtype).repeat_interleave(grid_thw.prod(dim=-1)).unsqueeze(1)
            )  # [L_total, 1]
            noise = noise * ns
        sample = (1 - timesteps_repeat) * original_samples + timesteps_repeat * noise
        sample = sample.to(original_samples.dtype, non_blocking=True)
        return sample, noise, timesteps

    def add_noise_dual(
        self,
        original_samples: torch.Tensor,
        grid_thw: torch.Tensor,
        force_zero: torch.Tensor,
        mask_ratio: float,
        noise_scale: torch.Tensor | None = None,
        min_t: float = 0.0,
        dual_item_mask: torch.Tensor | None = None,
        hi_band_mask: torch.Tensor | None = None,
        hi_band: tuple[float, float] | None = None,
        noise_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Self-Flow 的 Dual-Timestep Scheduling(arXiv:2603.06507 Sec 3.3/3.4)。

        每个 item 采两个同分布时间步 t,s ~ p(t);token 级 Bernoulli(mask_ratio) 掩码 M,
        τ_i = s(i∈M)/ t(否则);学生输入 x_τ 逐 token 加噪,教师输入 x_τmin 以
        τmin = min(t,s) 均匀加噪。两者共享同一份 noise(论文 Eq.6 的单一 x1)。
        t、s 都过完整的 p(t)(含 timeshift)→ 每 token 边际时间步分布不变,这是与
        diffusion forcing / full masking 的关键区别(论文 Fig 2b)。

        dual_item_mask: [N_items]，False 的图片令 s=t，退化为单时间步加噪。
        无论 mask 取值如何，仍采样 s 和 token mask，保持随机数消耗次数不变。

        Returns:
          x_student  : [L_total, C]  逐 token τ 加噪的学生输入 x_τ
          x_teacher  : [L_total, C]  τmin 均匀加噪的教师输入 x_τmin
          noise      : [L_total, C]  共享噪声(v-target = noise - x0 不变)
          tau_tok    : [L_total]     逐 token 时间步 τ(喂给模型的 per-token 条件)
          tmin_item  : [N_items]     每 item 的 τmin(教师的 per-item 标量条件)
        """
        assert 0.0 < mask_ratio <= 0.5, f"Self-Flow requires 0 < R_M <= 0.5, got {mask_ratio}"
        if hi_band_mask is not None and dual_item_mask is not None:
            assert not torch.any(hi_band_mask.bool() & dual_item_mask.bool()), (
                "hi_band_mask and dual_item_mask must be disjoint"
            )
        noise = torch.randn(
            size=original_samples.shape, device=original_samples.device,
            generator=self.generator, dtype=noise_dtype,
        )
        t = self.sample_train_timesteps(grid_thw, min_t=min_t, hi_band_mask=hi_band_mask, hi_band=hi_band)
        s = self.sample_train_timesteps(grid_thw, min_t=min_t)
        t[force_zero] = 0.0
        s[force_zero] = 0.0
        if dual_item_mask is not None:
            s = torch.where(dual_item_mask.to(torch.bool), s, t)

        lens = grid_thw.prod(dim=-1)
        u = torch.rand(original_samples.shape[0], generator=self.generator, device=self.generator.device)
        tok_in_mask = u < mask_ratio  # [L_total] — 论文的 u_i < R_M
        t_tok = t.repeat_interleave(lens)
        s_tok = s.repeat_interleave(lens)
        tau_tok = torch.where(tok_in_mask, s_tok, t_tok)  # [L_total]
        tmin_item = torch.minimum(t, s)  # [N_items]
        tmin_tok = tmin_item.repeat_interleave(lens)

        if noise_scale is not None:
            ns = noise_scale.to(noise.device, noise.dtype).repeat_interleave(lens).unsqueeze(1)
            noise = noise * ns

        tau_col = tau_tok.unsqueeze(1)  # [L_total, 1]
        tmin_col = tmin_tok.unsqueeze(1)
        x_student = ((1 - tau_col) * original_samples + tau_col * noise).to(original_samples.dtype, non_blocking=True)
        x_teacher = ((1 - tmin_col) * original_samples + tmin_col * noise).to(
            original_samples.dtype, non_blocking=True
        )
        return x_student, x_teacher, noise, tau_tok, tmin_item
