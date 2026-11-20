"""★ model_impl 的门面：加载一次，然后 `generate(prompt, think=True/False)`。

两段式（与 `tasks/transfusion/infer_thinkgen_dmd.py` 完全同一套配方）
--------------------------------------------------------------------
* `think=True`（ttg）：先用学生自己的 LM 做 KV-cache 自回归写思考链，停在 `<boi>` /
  `<|user|>`；再把「prompt + 思考链」当上下文走扩散段。
* `think=False`（direct / 纯 t2i）：跳过 AR 段，用训练里 t2i 那一路的布局直接扩散。

**扩散段两种模式走的是同一条代码路径**，区别只在送进去的 token 序列。

为什么扩散段直接复用 `tasks/transfusion/infer.py` 的函数
--------------------------------------------------------
`_denoise_and_decode` / `denoise_latents` 本身就是纯单机代码（零通信），而它是
现在出图那条链路**实跑验证过**的数值配方。重写一遍只会引入第二套口径——
renoise 的 x0 公式、force_zero 的参考行处理、VAE 解码的 rearrange 口径，
任何一处写歪都是"图能出、只是不对"。阶段二做 vendoring 时也是**原样复制**这两个函数。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from .loading import Runtime, load_runtime
from .specs import ModelSpec, get_spec


class _InferenceCollator:
    """逐行等价于 `veomni/models/transfusion/hy/inference.py:331-340 HYInferenceCollator`。

    自己写而不是 import：那个类 `from tasks.transfusion.train import MyCollator`，
    会把 4011 行的训练入口（dataset / optimizer / wandb）整条拖进来，而 `MyCollator`
    已经逐字抽进 `model_impl/pipeline.py` 了。
    """

    def __init__(self) -> None:
        from .pipeline import MyCollator

        self.collator = MyCollator()

    def __call__(self, features):
        batch = self.collator(features)
        batch["uncond_batch"] = self.collator([f["_uncond_feature"] for f in features])
        batch["idx"] = torch.tensor([int(f.get("idx", 0)) for f in features])
        batch["prompt"] = [f.get("prompt", "") for f in features]
        return batch


@dataclass
class Generation:
    """一次生成的产物。"""

    image: object                 # PIL.Image
    cot: str | None               # 思考链原文（think + final，think=False 时为 None）
    prompt: str
    #: ★ `</think>` 之后那一段 —— **唯一真正喂给扩散段的图像描述**。
    #: 判读几乎总看它（见 model_impl/tools/probe_cot_ablation.py 的实证），所以单独拿出来；
    #: 原先只存完整 CoT，想看"这次到底拿什么去画的"每次都得自己按 </think> 切。
    cot_final: str | None = None
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> str:
        self.image.save(path)
        return path


class Inferencer:
    """BAGEL 的 `InterleaveInferencer` 对位物。"""

    def __init__(self, runtime: Runtime) -> None:
        self.rt = runtime
        self.spec = runtime.spec
        self._transform = None
        self._collator = None
        self._decoder = None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_spec(cls, spec: str | ModelSpec, devices: list[int] | None = None, *,
                  ckpt: str | None = None, **kwargs) -> "Inferencer":
        """`ckpt` = 权重目录（含 `model.safetensors.index.json`）；不给就用
        `$VLLM_OMNI_GLM_IMAGE_WEIGHTS_ROOT/<spec 名>`。spec 决定 config 与采样配方，ckpt 只换权重。"""
        spec = get_spec(spec) if isinstance(spec, str) else spec
        if ckpt:
            import dataclasses

            spec = dataclasses.replace(spec, weights=os.path.abspath(os.path.expanduser(ckpt)))
        return cls(load_runtime(spec, devices, **kwargs))

    # ------------------------------------------------------------ 懒建的部件
    @property
    def transform(self):
        if self._transform is None:
            from .processing_hy import build_inference_transform

            self._transform = build_inference_transform(
                self.rt.tokenizer, self.rt.config, dict(self.spec.transform_kwargs), mode="ttg")
        return self._transform

    @property
    def collator(self):
        if self._collator is None:
            self._collator = _InferenceCollator()
        return self._collator

    @property
    def decoder(self):
        if self._decoder is None:
            from . import generation_hy as ar_decode

            self._decoder = ar_decode.HYTextDecoder(
                self.rt.model, self.rt.entry_device,
                shard_group=None,                       # 单进程：没有组内对齐这回事
                attn_impl=self.spec.ar["attn_impl"],
            )
        return self._decoder

    # ------------------------------------------------------------------ 生成
    @torch.no_grad()
    def generate(self, prompt: str, *, think: bool = True, size: tuple[int, int] = (1024, 1024),
                 seed: int = 1234, num_steps: int | None = None, cfg_scale: float | None = None,
                 sampler: str | None = None, idx: int = 0, ar_overrides: dict | None = None,
                 gen_ids: list[int] | None = None) -> Generation:
        """一条 prompt → 一张图。

        size 是 (height, width) 像素，**必须同时是 32（训练 scale_factor）与 VAE 步长的倍数**。
        `gen_ids` 允许外部塞一串现成的思考链 token（对拍用：喂同一串 token 才能只比扩散段）。
        """
        from . import generation_hy as ar_decode

        height, width = int(size[0]), int(size[1])
        steps = self.spec.num_steps if num_steps is None else int(num_steps)
        cfg = self.spec.cfg_scale if cfg_scale is None else float(cfg_scale)
        started = time.time()

        cot_ids: list[int] = []
        ar_info: dict = {}
        if think:
            if gen_ids is None:
                prefix, _user_text = self.transform.ar_prefix_ids(
                    prompt, height, width, mode="ttg", idx=idx,
                    with_res=self.spec.with_resolution_suffix)
                config = self._ar_config(seed, ar_overrides)
                raw, ar_info = self.decoder.generate(prefix, config, tokenizer=self.rt.tokenizer)
                cot_ids = ar_decode.trim_generated(raw)
                ar_info["model_grid"] = list(ar_decode.parse_model_grid(raw, self.rt.tokenizer) or ())
            else:
                cot_ids = list(gen_ids)
                ar_info = {"source": "caller"}
            ar_info["trimmed_len"] = len(cot_ids)
        t_ar = time.time()

        feature = self.transform.build_feature(
            prompt=prompt, height=height, width=width,
            mode=("ttg" if think else "t2i"), gen_ids=cot_ids or None,
            idx=idx, with_res=self.spec.with_resolution_suffix,
        )
        image_tensor = self._denoise(feature, steps=steps, cfg=cfg,
                                     sampler=sampler or self.spec.sampler, seed=seed)
        t_diff = time.time()

        return Generation(
            image=self._to_pil(image_tensor),
            cot=(self.rt.tokenizer.decode(cot_ids, skip_special_tokens=False) if think else None),
            cot_final=(ar_decode.split_final(cot_ids, self.rt.tokenizer) if think else None),
            prompt=prompt,
            meta={
                "spec": self.spec.name, "think": think, "height": height, "width": width,
                "seed": seed, "sampler": sampler or self.spec.sampler,
                "num_steps": steps, "cfg_scale": cfg,
                "ar": ar_info,
                # α：timeshift 的逐图对账量，与 8 卡 manifest 的 `alpha` 同一个算法
                "alpha": (round(self.rt.scheduler._glm_image_alpha(height, width), 4)
                          if hasattr(self.rt.scheduler, "_glm_image_alpha") else None),
                "input_tokens": int(feature["input_ids"].shape[0]),
                "n_prefix_tokens": feature["_n_prefix"],
                "ar_seconds": round(t_ar - started, 2),
                "diffusion_seconds": round(t_diff - t_ar, 2),
            },
        )

    # ------------------------------------------------------------ 内部实现
    def _ar_config(self, seed: int, overrides: dict | None):
        from .generation_hy import ARSamplingConfig

        values = dict(self.spec.ar)
        values.pop("attn_impl", None)
        values.update(seed=int(seed))
        values.update(overrides or {})
        return ARSamplingConfig(**values)

    def _denoise(self, feature: dict, *, steps: int, cfg: float, sampler: str, seed: int):
        """扩散段：原样调抽出来的 `_denoise_and_decode`（见模块 docstring）。"""
        from . import pipeline

        if cfg != 1.0:
            # HY 的已知静默 bug：cfg>1 时一个 batch 里混多个 prompt，约一半 prompt 出纹理噪声。
            # 这里天然 batch=1，留着这条断言是为了挡住将来加 batch 接口时忘了这件事。
            if len(feature.get("vae_latent_shape", [])) != 1:
                raise ValueError("cfg != 1 时每个 batch 只能有一个 prompt（HY 的已知静默 bug）")
        self.rt.scheduler.generator.manual_seed(int(seed))
        micro_batch = pipeline._batch_to_device(self.collator([feature]), self.rt.entry_device)
        images = pipeline._denoise_and_decode(
            micro_batch,
            model=self.rt.model, vae_model=self.rt.vae, noise_scheduler=self.rt.scheduler,
            model_config=self.rt.config,
            model_args=SimpleNamespace(timeinput_scaler=self.spec.timeinput_scaler),
            infer_args=SimpleNamespace(cfg_scale=cfg, sampling_num_steps=steps, sampler=sampler),
            model_runtime=SimpleNamespace(inference_kwargs={}),
            device=self.rt.entry_device,
        )
        return images[-1]

    @staticmethod
    def _to_pil(tensor):
        """[c, h, w]、值域 [-1, 1] → PIL。

        逐步照抄 `veomni/models/cogview/utils.py:261-271 save_images`
        （clamp((x+1)/2) → ×255 → numpy astype uint8，截断而非四舍五入）。
        对拍要逐像素比，这里换个等价写法都可能差 1。
        """
        import numpy as np
        from PIL import Image

        image = torch.clamp((tensor.float() + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        array = (255.0 * image.permute(1, 2, 0).numpy()).astype(np.uint8)
        return Image.fromarray(array)
