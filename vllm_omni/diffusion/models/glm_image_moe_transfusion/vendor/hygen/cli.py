#!/usr/bin/env python3
"""hygen 的命令行入口：出图 / 冒烟。

三种起法等价，挑顺手的：

    python -m hygen.cli --prompt "一只在雪地里打滚的柴犬" --think
    hygen --prompt "..." --think                  # pip install 之后的 console script
    bash scripts/run.sh --prompt "..." --think    # shell 启动器

**昇腾上会自己 bootstrap 算子包**（见 `_bootstrap_npu_ops`）：blockmask attention 与 rope
没有纯 torch 回退，而算子包是**进程级**的、必须在 `import torch_npu` 之前 source。
所以直接 `python` 起的时候，本入口会在真正建模之前**用 bash re-exec 自己一次**，
把 CANN 与算子包 source 进去。给了 `HYGEN_FA_VENDOR` 就自动做；没给就响亮报错并打印
那两行该跑的命令。GPU 上不需要这一步，直接跑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


#: 防止 re-exec 套娃
_BOOTSTRAP_FLAG = "HYGEN_OPS_BOOTSTRAPPED"


def _looks_like_npu() -> bool:
    """不 import torch 就判断这台机器是不是昇腾（import 了就晚了）。"""
    override = os.environ.get("HYGEN_DEVICE_TYPE") or os.environ.get("VEOMNI_DEVICE_TYPE")
    if override:
        return override == "npu"
    import shutil

    return shutil.which("npu-smi") is not None and shutil.which("nvidia-smi") is None


def _bootstrap_npu_ops() -> None:
    """昇腾上把算子包 source 进来 —— 做法是**用 bash re-exec 自己一次**。

    为什么只能这样：`ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH` 是 CANN **加载算子库时**
    读的，`import torch_npu` 之后在 python 里改已经没用了。而 blockmask attention 与 rope
    在本库里没有纯 torch 回退（eager 分支是空函数、rope 的 fused 入口默认 raise）。

    已经 source 过（`ASCEND_CUSTOM_OPP_PATH` 里有 custom_transformer）就什么都不做 ——
    所以经 `scripts/run.sh` 起时这段是空转。
    """
    if os.environ.get(_BOOTSTRAP_FLAG) or not _looks_like_npu():
        return
    if "custom_transformer" in os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""):
        return                      # 已经由 run.sh 或用户自己 source 过
    vendor = os.environ.get("HYGEN_FA_VENDOR", "")
    cann = os.environ.get("ASCEND_SET_ENV", "/usr/local/Ascend/ascend-toolkit/set_env.sh")
    if not vendor:
        raise SystemExit(
            "[hygen] ✗ 昇腾上必须有 patched FA 算子包，但 HYGEN_FA_VENDOR 没设。\n"
            "        blockmask attention 与 rope 没有纯 torch 回退，而算子包是**进程级**的，\n"
            "        必须在 import torch_npu 之前 source ⇒ 本入口会自己 re-exec 处理，\n"
            "        但要你先告诉它包在哪：\n"
            "            export HYGEN_FA_VENDOR=<包>/vendors/custom_transformer\n"
            "        ⚠ 910B 与 910C/910_93 的包**不通用**。给错轻则崩 161001，\n"
            "          重则 mask 被 stock tiling 接受、prefix 被忽略 ⇒ 静默退化成 causal。")
    env_script = os.path.join(vendor, "bin", "set_env.bash")
    if not os.path.isfile(env_script):
        raise SystemExit(f"[hygen] ✗ 读不到 {env_script}（HYGEN_FA_VENDOR 指错了？）")
    # 剔掉可能已有的别的 FA 路径，免得混包；再 source，再把自己原样起回来
    strip = ("export ASCEND_CUSTOM_OPP_PATH=$(tr ':' '\\n' <<< \"${ASCEND_CUSTOM_OPP_PATH:-}\" "
             "| grep -v custom_transformer | paste -sd: -); "
             "export LD_LIBRARY_PATH=$(tr ':' '\\n' <<< \"${LD_LIBRARY_PATH:-}\" "
             "| grep -v custom_transformer | paste -sd: -); ")
    command = (f"{strip}"
               f"[ -r {cann!r} ] && source {cann!r}; "
               f"source {env_script!r}; "
               f"exec {sys.executable!r} " + " ".join(repr(a) for a in sys.argv))
    print(f"[hygen] bootstrap 算子包 → re-exec（vendor={vendor}）", flush=True)
    os.environ[_BOOTSTRAP_FLAG] = "1"
    os.execvp("bash", ["bash", "-lc", command])


def main() -> None:
    _bootstrap_npu_ops()
    parser = argparse.ArgumentParser(description="hygen 推理")
    parser.add_argument("--ckpt", "--weights", dest="ckpt", default=None,
                        help="权重目录（含 model.safetensors.index.json 与 model-*.safetensors）。"
                             "不给就用 $HYGEN_WEIGHTS_ROOT/<spec 名>")
    parser.add_argument("--spec", default="dmd2_lyzsft10000_s2000",
                        help="配方（config + 采样参数）。默认就是少步学生，正常不用改")
    parser.add_argument("--devices", default="0,1,2,3", help="逗号分隔的卡号（NPU / GPU 都是这个）")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None, help="一行一条；与 --prompt 二选一")
    parser.add_argument("--think", dest="think", action="store_true", default=None)
    parser.add_argument("--no-think", dest="think", action="store_false")
    parser.add_argument("--both", action="store_true", help="同一条 prompt 跑 think 与 no-think 两版")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--seed", type=int, default=1234,
                        help="噪声种子**基准**。默认按 8 卡链路同一套公式派生："
                             "seed = base + 题号*1009，所以同一题跨轮可复现、不同题各有噪声。")
    parser.add_argument("--fixed-seed", action="store_true",
                        help="所有 prompt 共用 --seed（只在做「同噪声不同 prompt」的消融时才要）")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--sampler", default=None, choices=["renoise", "ode"],
                        help="覆盖采样器。默认取 spec 的值：学生线是 renoise（SDE，蒸馏的就是这个"
                             "口径），teacher 线是 ode。⚠ renoise 每步重新加噪用的是**全局 RNG**"
                             "（没传 generator）⇒ 同种子也不保证可复现；要确定性/要对拍就用 ode。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="覆盖思考链长度上限（冒烟用）")
    parser.add_argument("--out", default="hygen_out")
    parser.add_argument("--smoke", action="store_true", help="只加载模型 + 跑一次最小前向，不出图")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from hygen.inferencer import Inferencer

    devices = [int(d) for d in args.devices.split(",") if d.strip() != ""]
    height, _, width = args.size.partition("x")
    height, width = int(height), int(width)

    t0 = time.time()
    inferencer = Inferencer.from_spec(args.spec, devices, ckpt=args.ckpt)
    print(f"[hygen] 加载用时 {time.time() - t0:.1f}s", flush=True)

    if args.smoke:
        _smoke(inferencer, height, width)
        return

    # 读法与 8 卡链路逐字一致（`infer_thinkgen_dmd.py:_read_prompts`）：
    # **行号 = 题号**，含空行与被注释掉的行。这样"把不想跑的行注释掉"就能挑子集而题号不变，
    # 跨轮、跨链路对账都对得上 —— 题号直接决定噪声种子，错位就比不了图。
    prompts: list[tuple[int, str]] = []
    if args.prompts_file:
        with open(args.prompts_file) as handle:
            for line_no, line in enumerate(handle):
                text = line.strip()
                if text and not text.startswith("#"):
                    prompts.append((line_no, text))
    if args.prompt:
        prompts.append((len(prompts), args.prompt))
    if not prompts:
        raise SystemExit("[fatal] 需要 --prompt 或 --prompts-file")

    modes = [True, False] if args.both else [True if args.think is None else args.think]
    os.makedirs(args.out, exist_ok=True)
    overrides = {"max_new_tokens": args.max_new_tokens} if args.max_new_tokens else None
    for index, prompt in prompts:
        # 与 8 卡链路同一个公式（`Task.noise_seed`）：base + 题号*1009 + sample(=0)。
        seed = args.seed if args.fixed_seed else args.seed + index * 1009
        for think in modes:
            started = time.time()
            result = inferencer.generate(
                prompt, think=think, size=(height, width), seed=seed, idx=index,
                num_steps=args.steps, cfg_scale=args.cfg, sampler=args.sampler,
                ar_overrides=overrides,
            )
            tag = "ttg" if think else "direct"
            stem = os.path.join(args.out, f"{index:03d}_{tag}_{height}x{width}_s{seed}")
            result.save(stem + ".png")
            with open(stem + ".json", "w") as handle:
                json.dump({**result.meta, "prompt": prompt, "cot": result.cot,
                           "cot_final": result.cot_final},
                          handle, ensure_ascii=False, indent=2)
            if result.cot_final:
                # final 段单独一个文件：它是唯一喂给扩散段的描述，判读时最常看
                with open(stem + ".final.txt", "w") as handle:
                    handle.write(result.cot_final)
            print(f"[hygen] [{tag}] idx={index} {time.time() - started:.1f}s "
                  f"ar={result.meta['ar_seconds']}s diff={result.meta['diffusion_seconds']}s "
                  f"cot_len={result.meta['ar'].get('trimmed_len', 0)} -> {stem}.png", flush=True)
            _refresh_viewer(args.out)   # 每出一张就刷，中途中断也有得看


def _refresh_viewer(out_dir: str) -> None:
    """出一张刷一次 viewer。

    采完即进 viewer 是这个项目的一贯要求：跑一轮要几分钟到几十分钟，
    等全跑完再拼 html 的话，中途想看一眼就得自己去翻 png。
    viewer 坏了不该连累出图，所以这里吞掉异常、只打一行。
    """
    try:
        from hygen.tools.build_viewer import collect  # noqa: F401  （只为确认可导入）
        import subprocess
        import sys as _sys

        tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "build_viewer.py")
        subprocess.run([_sys.executable, tool, out_dir], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as error:  # noqa: BLE001
        print(f"[hygen] ⚠ viewer 刷新失败（不影响出图）：{error}", flush=True)


def _smoke(inferencer, height: int, width: int) -> None:
    """最小前向：不跑 AR、只过一次扩散段的第一步，验管线通不通。"""
    import torch

    transform = inferencer.transform
    feature = transform.build_feature(prompt="冒烟", height=height, width=width, mode="t2i")
    print(f"[hygen] smoke: input_ids={feature['input_ids'].shape[0]} tokens, "
          f"grid={feature['vae_latent_shape'][0].tolist()}", flush=True)
    started = time.time()
    image = inferencer._denoise(feature, steps=1, cfg=1.0, sampler=inferencer.spec.sampler, seed=0)
    print(f"[hygen] smoke: 1 步扩散 + VAE 解码 OK，{time.time() - started:.1f}s，"
          f"image={tuple(image.shape)} dtype={image.dtype} "
          f"finite={bool(torch.isfinite(image).all())}", flush=True)
    for device in inferencer.rt.devices:
        from hygen.base import get_device_type

        _dev = getattr(torch, get_device_type(), None)
        _peak = _dev.max_memory_allocated(device) / 2**30 if _dev is not None else 0.0
        print(f"[hygen] smoke: {get_device_type()}{device} peak={_peak:.1f}GiB")


if __name__ == "__main__":
    main()
