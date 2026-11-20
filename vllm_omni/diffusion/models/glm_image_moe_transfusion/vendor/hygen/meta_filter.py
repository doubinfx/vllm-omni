# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
按 ``meta.jsonl`` 里的字段阈值过滤样本（源级别开关，默认关）。

动机：wds 的 *.meta.jsonl 里有一堆离线算好的分数（``portrait_p_keep`` /
``score_info.watermark_prob_siglip2`` / ``score_info.sharpness`` …），想按阈值
提纯某一源时，以前只能离线重切一份数据集。这里把阈值判据搬到 transform 里，
yaml 改一行就能生效：

    transform_kwargs:
      meta_filters:
      - key: portrait_p_keep
        op: '>'            # > >= < <= == != （省略默认 '>'）
        value: 0.4
        on_missing: drop   # 这一条判据取不到值时 drop / keep，省略则继承下面这个
      meta_filter_on_missing: keep   # 全局缺省策略，不写是 keep（= 不认识的行照常训）

语义：
  • 多条判据之间是 **与**（全过才留）。
  • ``key`` 先按字面量在 meta 顶层找；找不到再按 ``.`` 逐层下钻
    （``score_info.sharpness``），每层同样先试字面量 —— 所以 key 里本身带点的
    字段（``..._zh.cn_long``）也能命中。
  • 取不到值（键缺失 / 值是 null / 值不是数且判据要求数）→ 走 ``on_missing``。
  • 被判掉的样本以 ValueError 抛出，消息里带 ``meta_filter`` 前缀，
    dataset.py 的 ``_SILENT_SKIP_SUBSTRINGS`` 认这个前缀 ⇒ 静默跳过、不刷 warning。

⚠ 过滤是 **采样之后** 发生的：源的 ``weight`` 不会自动补偿被丢掉的比例，所以通过率
  低的判据会让这一源的实际占比等比例缩水（丢 40% 就等于 weight ×0.6）。要维持配比
  得自己把 weight 调上去。
"""

from __future__ import annotations

import operator
from typing import Any, NamedTuple

from . import base as logging


logger = logging.get_logger(__name__)


_MISSING = object()

# 判据算子。符号和英文名都收，yaml 里 '>' 需要引号，写 gt 可以省掉。
_OPS = {
    ">": operator.gt,
    "gt": operator.gt,
    ">=": operator.ge,
    "ge": operator.ge,
    "<": operator.lt,
    "lt": operator.lt,
    "<=": operator.le,
    "le": operator.le,
    "==": operator.eq,
    "eq": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
}
# == / != 允许拿字符串、布尔直接比，其余算子一律数值比较。
_EQ_OPS = ("==", "eq", "!=", "ne")
_ON_MISSING = ("drop", "keep")
_ALLOWED_FIELDS = ("key", "op", "value", "on_missing")


class MetaFilterRule(NamedTuple):
    key: str
    op: str
    fn: Any
    value: Any
    on_missing: str

    def describe(self) -> str:
        return f"{self.key} {self.op} {self.value!r}"


def compile_meta_filters(specs: Any, default_on_missing: str = "keep") -> tuple[MetaFilterRule, ...]:
    """把 yaml 里的 dict 列表校验成规则元组。构造 transform 时调一次，
    让写错的 key/op 在建 dataloader 时就炸，而不是等到 worker 里逐条样本炸。"""
    if not specs:
        return ()
    if default_on_missing not in _ON_MISSING:
        raise ValueError(f"meta_filter_on_missing 只能是 {_ON_MISSING}，得到 {default_on_missing!r}")
    if isinstance(specs, dict):  # 只写一条判据时允许不套列表
        specs = [specs]
    if not isinstance(specs, (list, tuple)):
        raise ValueError(f"meta_filters 要是 dict 的列表，得到 {type(specs).__name__}")

    rules = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"meta_filters 的每一项要是 dict，得到 {spec!r}")
        unknown = set(spec) - set(_ALLOWED_FIELDS)
        if unknown:
            raise ValueError(f"meta_filters 里有不认识的字段 {sorted(unknown)}，只支持 {_ALLOWED_FIELDS}")
        key = spec.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"meta_filters 的 key 要是非空字符串，得到 {key!r}")
        op = str(spec.get("op", ">"))
        if op not in _OPS:
            raise ValueError(f"meta_filters[{key}] 的 op={op!r} 不支持，可选 {sorted(_OPS)}")
        if "value" not in spec:
            raise ValueError(f"meta_filters[{key}] 缺 value")
        value = spec["value"]
        if op not in _EQ_OPS:
            value = _as_number(value)
            if value is _MISSING:
                raise ValueError(f"meta_filters[{key}] 的 op={op!r} 要数值阈值，得到 {spec['value']!r}")
        on_missing = str(spec.get("on_missing", default_on_missing))
        if on_missing not in _ON_MISSING:
            raise ValueError(f"meta_filters[{key}] 的 on_missing 只能是 {_ON_MISSING}，得到 {on_missing!r}")
        rules.append(MetaFilterRule(key=key, op=op, fn=_OPS[op], value=value, on_missing=on_missing))
    return tuple(rules)


def _as_number(value: Any) -> Any:
    """数值化，失败返回 ``_MISSING``。bool 走 True=1/False=0（``portrait_keep``
    这种布尔字段就能用 ``> 0.5`` 判）；数字字符串也认。"""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return _MISSING
    return _MISSING


def lookup_meta_value(meta: Any, key: str) -> Any:
    """取 ``key`` 对应的值，取不到返回 ``_MISSING``。每层先试字面量再按 ``.`` 下钻。"""
    node = meta
    remaining = key
    while True:
        if not isinstance(node, dict):
            return _MISSING
        if remaining in node:
            return node[remaining]
        if "." not in remaining:
            return _MISSING
        head, remaining = remaining.split(".", 1)
        if head not in node:
            return _MISSING
        node = node[head]


def check_meta_filters(meta: Any, rules: tuple[MetaFilterRule, ...], where: str = "") -> None:
    """逐条判据过一遍 meta，判掉就抛 ValueError（上层 dataset 静默跳过该样本）。"""
    if not rules:
        return
    for rule in rules:
        raw = lookup_meta_value(meta, rule.key)
        missing_why = ""
        if raw is _MISSING:
            missing_why = "这一行没有这个字段"
        elif raw is None:
            missing_why = "字段值是 null"
        else:
            if rule.op in _EQ_OPS and not isinstance(rule.value, (int, float)):
                if not rule.fn(raw, rule.value):
                    raise ValueError(f"meta_filter: {where}{rule.describe()} 不满足（实际 {raw!r}）")
                continue
            number = _as_number(raw)
            if number is _MISSING:
                missing_why = f"字段值 {raw!r} 不是数值"
            elif not rule.fn(number, rule.value):
                raise ValueError(f"meta_filter: {where}{rule.describe()} 不满足（实际 {number}）")
            else:
                continue

        # 取不到判据 —— 按这条规则自己的 on_missing 处理。
        logger.warning_once(
            f"meta_filters 取不到判据：{rule.key!r} —— {missing_why}。按 on_missing={rule.on_missing} 处理。"
            "⚠ 这条是 warning_once：每个 dataloader worker 每个 key 只打一次，且只有 LOCAL_RANK=0 的进程打 ⇒ "
            "它只说明「出现过」，**读不出缺失率**；要缺失率得离线扫 meta.jsonl。"
        )
        if rule.on_missing == "drop":
            raise ValueError(f"meta_filter: {where}{rule.key} 取不到判据（{missing_why}），on_missing=drop")
