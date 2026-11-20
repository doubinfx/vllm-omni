"""model_impl —— HY-30B 的独立推理包（CoT 生图 / 纯 T2I）。

形态对标 BAGEL 的推理仓库：加载一次、常驻、`Inferencer.generate(prompt, think=...)`。
与 `tasks/transfusion/infer_thinkgen_dmd.py` 那条 8 卡批跑链路的关系：
**那条路一行不改，它是本包的对拍参照系**；本包是同一套数值配方的单进程封装。

布局按 transformers 的习惯：

    configuration_hy.py      config 类
    modeling_hy.py           模型主体（+ _text / _vision / _vae 几个分文件）
    processing_hy.py         token 布局（只有 eval 需要的那部分）
    generation_hy.py         AR 解码
    pipeline.py              扩散采样 + collator
    inferencer.py            对外的 Inferencer

所有 modeling/processing 代码都由 `model_impl/tools/gen_*.py` 从 veomni **逐字抽取**生成，
函数体一字未改；每个生成器都有 `--check`（上游漂移）和 `--verify`（逐键/字节码/逐层）。
"""

from .specs import SPECS, ModelSpec, get_spec  # noqa: F401,E402

__all__ = ["SPECS", "ModelSpec", "get_spec", "Inferencer"]


def __getattr__(name):
    # Inferencer 会拉起整条 veomni import 链（几秒），不该在 `import model_impl` 时就付这个代价。
    if name == "Inferencer":
        from .inferencer import Inferencer

        return Inferencer
    raise AttributeError(name)
