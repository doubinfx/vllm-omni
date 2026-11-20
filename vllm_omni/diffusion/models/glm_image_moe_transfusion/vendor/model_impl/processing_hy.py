"""【自动生成，勿手改】token 布局与 collator —— **只有 eval 需要的那部分**。

由 `model_impl/tools/gen_processing.py` 生成。训练数据管线（webdataset 读取、caption_pool
采样、图像增强、res_sched、各 format 分支）**没有搬**，因为 eval 用不到
（用户 2026-11-17 的约束：只做 eval 的适配）。

对外主要是 `build_inference_transform(tokenizer, config, transform_kwargs, mode=...)`
—— 它造出 `ThinkGenInferenceTransform`，`build_feature()` 把 prompt + 分辨率
（+ 可选的思考链 token）变成送进扩散段的那一条 packed 序列。

⚠ token 布局错了是**静默**的（图照出、只是不对），所以改动这里之后必须跑
`python3 model_impl/tools/check_layout.py samples_ttg/<某个 8 卡 run>`。
"""

from __future__ import annotations

# ── import 区由 collect_imports() 自动筛出 ──
import json
import os
import re
from typing import Callable, Optional
import io
import math
import random
from dataclasses import dataclass, field, fields
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor
from transformers import AutoTokenizer
from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl import base as logging
from .meta_filter import check_meta_filters, compile_meta_filters
import functools
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torch.utils.data._utils.collate import default_collate, default_collate_fn_map


from . import base as _glm_image_base
logger = _glm_image_base.get_logger(__name__)

# ══════════ 搬自 veomni/data/transforms/thinkgen_templates.py（整文件） ══════════

T2I_REQUEST_TEMPLATES = [
    "Generate an image based on the following description:",
    "Create an image based on this text:",
    "Produce an artwork based on the description below:",
    "Visualize the scene described in the following text:",
    "Generate a visual representation of this description:",
    "Create a realistic image based on these details:",
    "Illustrate the scene described here:",
    "Make an image that corresponds to this description:",
    "Render an image based on the following prompt:",
    "Design a visual interpretation of this text:",
    "Construct an image based on the provided details:",
    "Synthesize a picture based on the following narrative:",
    "Generate an image from this detailed description:",
    "Create a digital painting based on the text below:",
    "Produce a visual based on the following scenario:",
    "Visualize the following:",
    "Depict the following scene:",
    "Illustrate the following scenario:",
    "Show me an image of the following:",
    "Create a picture of the following scene:",
    "Generate an image matching this description:",
    "Paint a scene described as follows:",
    "Here is a description, please generate an image:",
    "Please visualize the following detailed description:",
    "Render the following scene:",
    "Create an illustration of the following concept:",
    "Draw a scene matching the following details:",
    "Generate a view of the following:",
    "Interpret the following description visually:",
    "Capture the essence of the following text in an image:",
    "Generate an image based on the following description.",
    "Create an image that matches this text.",
    "Please generate an image according to the text below.",
    "Visualize the scene described in the following text.",
    "Produce an artwork based on this description.",
    "Create a realistic image from these details.",
    "Generate a visual representation of this description.",
    "Illustrate the scene described here.",
    "Make an image that corresponds to this description.",
    "Render an image based on the following prompt.",
    "Design a visual interpretation of this text.",
    "Construct an image based on the provided details.",
    "Synthesize a picture based on the following narrative.",
    "Generate an image from this detailed description.",
    "Create a digital painting based on the text below.",
    "Produce a visual based on the following scenario.",
    "Turn this description into an image.",
    "Use this text to generate an image.",
    "Create an illustration for the following text.",
    "Please draw the scene described below.",
    "Here is a description for an image generation task.",
    "The following is a description of an image.",
    "I would like you to generate an image based on this text.",
    "Here is a prompt to visualize.",
    "This is a detailed description of a scene.",
    "Below is a text description for image creation.",
    "Here is the caption for the image.",
    "The text below describes the image I want.",
    "I need a visualization of the following description.",
    "Here is a scenario to depict.",
    "The following text describes a visual scene.",
    "Use the following details to create an artwork.",
    "This text describes the visual style and content.",
    "I have a detailed description for you to visualize.",
    "The following paragraph describes an image.",
    "Here are the details for the image generation.",
    "Use the text below as the image prompt.",
    "This is a request to generate a specific image.",
    "Below is the narrative for the illustration.",
    "Here is the concept art description.",
    "Generate an image of",
    "Create an image of",
    "Make a picture of",
    "Produce an image showing",
    "Generate a visual of",
    "Create a visual depiction of",
    "Make an artwork of",
    "Produce a picture of",
    "Generate a photo of",
    "Create a photograph of",
    "Make a drawing of",
    "Generate an illustration of",
    "Create a digital painting of",
    "Make a sketch of",
    "Generate a 3D render of",
    "Create a realistic image of",
    "Produce a high-quality image of",
    "Generate a scene showing",
    "Create a design for",
    "Make a graphic of",
    "Render an image of",
    "Output an image of",
    "Construct a visual of",
    "Compose a picture of",
    "Synthesize an image of",
    "Please generate an image of",
    "Could you generate an image of",
    "Please create a picture of",
    "Can you draw a",
    "I would like an image of",
    "Please make a visual of",
    "Help me draw a",
    "Can you help me create an image of",
    "Please paint a",
    "I need a picture of",
    "Could you make an illustration of",
    "Please show me an image of",
    "I want to see a picture of",
    "Can you visualize",
    "Please design a",
    "I'd like to request an image of",
    "Would you kindly draw",
    "Please sketch a",
    "Can you produce a photo of",
    "Help me visualize",
    "Please render a scene of",
    "I want an artwork showing",
    "Can you generate a background of",
    "Please create a character showing",
    "I need a visual representation of",
]

THINKING_TEMPLATE = (
    "Generate an image based on the user's request. Before generating the image, "
    "carefully analyze the request, plan the composition, visual elements, style, "
    "and constraints, and resolve any ambiguities or inconsistencies. Explicitly "
    "present your analysis and generation plan before producing the final image:"
)

__all__ = ["T2I_REQUEST_TEMPLATES", "THINKING_TEMPLATE"]


# ══════════ 搬自 veomni/data/transforms/lyz_quote.py（整文件） ══════════

__all__ = [
    "QuoteUnbalanced",
    "split_quotation_legacy",
    "split_quotation_stack",
    "split_structured_json",
    "resolve_splitter",
    "quote_stray_hits",
    "QUOTE_MODES",
]

# ── veomni/data/transforms/lyz_quote.py:61-107 ──
def split_quotation_legacy(prompt: str, quote_pairs: Optional[list[tuple[str, str]]] = None):
    # luyizhuo 0913 修复（默认开；VEOMNI_LEGACY_SPLIT_QUOTATION=1 回退旧行为以复现旧 run）：
    #   ① 词内撇号保护原正则只认 ASCII 撇号(U+0027)，缩写常用弯撇号 U+2019 漏保护
    #      → 两个落单弯撇号跨词互相配对，把普通散文误判为引号、逐字符编码、token 虚高。
    #   ② 占位符原为 "longcat_$##$_longcat"*(i+1)，第 i 个是第 i+1 个的前缀子串，
    #      还原时先还原短的会破坏长的（实测产生 don't→doesn'tdoesn't 脏文本）。
    _legacy = os.environ.get("VEOMNI_LEGACY_SPLIT_QUOTATION") == "1"
    if _legacy:
        word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    else:
        word_internal_quote_pattern = re.compile("[a-zA-Z]+['’][a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote: list[list[str]] = []

    _uniq = set(matches_word_internal_quote_pattern) if _legacy else dict.fromkeys(
        matches_word_internal_quote_pattern
    )
    for i, word_src in enumerate(_uniq):
        word_tgt = "longcat_$##$_longcat" * (i + 1) if _legacy else f"longcat{i}_$##$_longcat"
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
        quotes = ["'", '"', "‘", "’", "“", "”"]
        for q1 in quotes:
            for q2 in quotes:
                if (q1, q2) not in quote_pairs:
                    quote_pairs.append((q1, q2))

    pattern = "|".join(
        [re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2) for q1, q2 in quote_pairs]
    )

    parts = re.split(f"({pattern})", prompt)

    result: list[tuple[str, bool]] = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result

_SQ_LETTER = r"a-zA-ZÀ-ɏ̀-ͯͰ-ϿЀ-ӿ"

_SQ_WORD_INTERNAL = re.compile(rf"(?<=[{_SQ_LETTER}])['’](?=[{_SQ_LETTER}])")

_SQ_WORD_HYPHEN = re.compile(rf"(?<=[{_SQ_LETTER}])['’](?=-[{_SQ_LETTER}])")

_SQ_PAIRS = (
    ("'", "'"),
    ('"', '"'),
    ("‘", "’"),
    ("“", "”"),
    ("„", "”"),
    ("„", "“"),
    ("‚", "’"),
    ("‚", "‘"),
)

_SQ_OPENERS = frozenset(q1 for q1, _ in _SQ_PAIRS)

_SQ_CLOSERS: dict[str, frozenset] = {}

for _q1, _q2 in _SQ_PAIRS:
    _SQ_CLOSERS[_q2] = _SQ_CLOSERS.get(_q2, frozenset()) | {_q1}

_SQ_CHARS = re.compile("[" + re.escape("".join(sorted(_SQ_OPENERS | set(_SQ_CLOSERS)))) + "]")

_QUOTE_STRAY_HITS = 0

# ── veomni/data/transforms/lyz_quote.py:141-142 ──
def quote_stray_hits() -> int:
    return _QUOTE_STRAY_HITS

# ── veomni/data/transforms/lyz_quote.py:145-211 ──
def split_quotation_stack(prompt: str, tolerant: bool = False):
    """按类型配对的栈匹配引号切分。

    Args:
        tolerant: True = 落单引号当普通字符忽略，已匹配上的片段照常返回。
            False = 严格，不平衡时返回 None。

    Returns:
        list[tuple[str, bool]]：matched 段**含首尾引号**（与旧 run 一致）。
        None：仅 tolerant=False 且引号不平衡时。
    """
    global _QUOTE_STRAY_HITS
    if not prompt or not _SQ_CHARS.search(prompt):
        return [(prompt, False)] if prompt else []
    skip = frozenset()
    if "'" in prompt or "’" in prompt:
        skip = {m.start() for m in _SQ_WORD_INTERNAL.finditer(prompt)} | {
            m.start() for m in _SQ_WORD_HYPHEN.finditer(prompt)
        }
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    stray = 0
    for m in _SQ_CHARS.finditer(prompt):
        i = m.start()
        if i in skip:
            continue
        ch = m.group()
        hit = None
        if ch in _SQ_CLOSERS:
            can = _SQ_CLOSERS[ch]
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] in can:
                    hit = j
                    break
        if hit is not None:
            spans.append((stack[hit][1], i))
            del stack[hit:]  # 弹出匹配项及其上方所有未闭合的开引号
        elif ch in _SQ_OPENERS:
            stack.append((ch, i))
        else:
            # 只能作闭引号的字符却无配对（多余闭引号）
            if not tolerant:
                return None
            stray += 1
    stray += len(stack)  # 扫完仍未闭合的开引号
    if stray:
        if not tolerant:
            return None
        _QUOTE_STRAY_HITS += 1
    # 只保留最外层；空内容片段（"" / ''）不输出
    spans.sort()
    keep: list[tuple[int, int]] = []
    last = -1
    for a, b in spans:
        if a > last and b > a + 1:
            keep.append((a, b))
            last = b
    out: list[tuple[str, bool]] = []
    pos = 0
    for a, b in keep:
        if a > pos:
            out.append((prompt[pos:a], False))
        out.append((prompt[a : b + 1], True))
        pos = b + 1
    if pos < len(prompt):
        out.append((prompt[pos:], False))
    return out

# ── veomni/data/transforms/lyz_quote.py:214-216 ──
class QuoteUnbalanced(ValueError):
    """final prompt 的引号不平衡 —— 由 transform 抛出，调用方（dataset）跳该样本。
    消息串含 'quote-unbalanced'。"""

_JSON_TEXT_FIELD = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')

# ── veomni/data/transforms/lyz_quote.py:226-262 ──
def split_structured_json(prompt: str):
    """只有 text_render 的 text 值为 True。返回 None = "不是我们认得的结构"，
    调用方降级为整段普通编码（= "off"），**绝不退回 legacy**（那是 12 倍过抽）。

    三道闸：① 整条能 json.loads 且是 dict 带 list 型 text_render；
            ② 正则扫到的 `"text": "…"` 个数 == text_render 里带 text 的条数；
            ③ 逐个 JSON 解码后与 text_render[i]["text"] 逐字相等（顺序也一致）。
    """
    try:
        obj = json.loads(prompt)
    except Exception:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("text_render"), list):
        return None
    want = [str(t["text"]) for t in obj["text_render"] if isinstance(t, dict) and "text" in t]
    hits = list(_JSON_TEXT_FIELD.finditer(prompt))
    if len(hits) != len(want):
        return None
    for m, w in zip(hits, want):
        try:
            if json.loads('"%s"' % m.group(1)) != w:
                return None
        except Exception:
            return None

    out: list[tuple[str, bool]] = []
    pos = 0
    for m in hits:
        a, b = m.start(1), m.end(1)
        if a > pos:
            out.append((prompt[pos:a], False))
        if b > a:  # 空串不产生片段
            out.append((prompt[a:b], True))
        pos = b
    if pos < len(prompt):
        out.append((prompt[pos:], False))
    return out

QUOTE_MODES = ("legacy", "stack", "stack_tolerant", "stack_strict", "json_textrender", "off")

# ── veomni/data/transforms/lyz_quote.py:268-298 ──
def resolve_splitter(quote_mode: str) -> tuple[bool, Optional[Callable]]:
    """quote_mode → 新仓 `encode_prompt` 的 (split_quoted_text, quote_splitter)。

    `split_quoted_text=False` 时上游直接走 `[(prompt, False)]`，splitter 不被调用。
    """
    if quote_mode == "off":
        return (False, None)
    if quote_mode == "legacy":
        return (True, split_quotation_legacy)
    if quote_mode == "stack_tolerant":
        return (True, lambda s: split_quotation_stack(s, tolerant=True))
    if quote_mode == "json_textrender":

        def _json(s):
            parts = split_structured_json(s)
            return [(s, False)] if parts is None else parts

        return (True, _json)
    if quote_mode in ("stack", "stack_strict"):
        strict = quote_mode == "stack_strict"

        def _stack(s):
            parts = split_quotation_stack(s)
            if parts is None:  # 不平衡
                if strict:
                    raise QuoteUnbalanced(f"quote-unbalanced in final prompt: {s[:60]!r}")
                return [(s, False)]  # 降级：整段普通编码
            return parts

        return (True, _stack)
    raise ValueError(f"unknown quote_mode {quote_mode!r}; 可选 {QUOTE_MODES}")


# ══════════ 搬自 veomni/data/transforms/wds_transform.py（闭包 18/20 个定义） ══════════

logger = logging.get_logger(__name__)

USER_TOKEN_ID = 154827  # <|user|>

ASSISTANT_TOKEN_ID = 154828  # <|assistant|>

GEN_IMAGE_REQUEST_TOKEN_ID = 154857

_CALICO_STRUCTURE_CAPTION_KEYS = frozenset(
    f"calico_structure_{version}_{lang}_qwen37_{variant}"
    for version in ("v1", "v2")
    for lang in ("zh", "en")
    for variant in ("str", "refine")
)

_STRUCTURE_TEXT_START = "\ue000"

_STRUCTURE_TEXT_END = "\ue001"

_STRUCTURE_QUOTE_PAIRS: tuple[tuple[str, str], ...] = ((_STRUCTURE_TEXT_START, _STRUCTURE_TEXT_END),)

# ── veomni/data/transforms/wds_transform.py:69-97 ──
def _compact_json_string(value: str, mark_text_render: bool = False) -> str:
    """把结构 caption 压成单行 JSON，顺带删掉 type、给要渲染的文字打上哨兵。

    - 压缩：原始数据是 indent 过的（p50 2000+ 字符里一大半是换行和缩进空格），
      ``separators=(",", ":")`` 去掉格式化空白，字段一个不少。
    - 删 type：``type`` 是出库时的数据集标签（``structure_v1_ct_photography`` 之类），
      不是图像/布局信息，不该出现在 t2i prompt 里。
    - 打哨兵：``text_render[*].text`` 是图上要渲染的文字，下游 ``encode_prompt``
      对哨兵内的内容逐字符 tokenize（字形监督），其余部分照常 BPE。

    ``mark_text_render=True`` 时结构不符（不是 dict / 没有 text_render 列表）直接抛，
    由调用方记日志 —— 这类 caption 整条就是 prompt，结构不对没法「退化成普通文本」用。
    """
    parsed = json.loads(value)
    if mark_text_render and not isinstance(parsed, dict):
        raise ValueError("Calico structure caption is not a JSON object")
    if isinstance(parsed, dict):
        parsed.pop("type", None)
        if mark_text_render:
            text_render = parsed.get("text_render")
            if not isinstance(text_render, list):
                raise ValueError("Calico structure caption has no text_render list")
            for text_item in text_render:
                if isinstance(text_item, dict) and isinstance(text_item.get("text"), str):
                    # 兜底：文字里万一自带哨兵码位会破坏配对（strict 模式直接判不平衡
                    # 抛错）。PUA 码位没有字形、不可能是真的 prompt 内容，先清掉。
                    text = text_item["text"].replace(_STRUCTURE_TEXT_START, "").replace(_STRUCTURE_TEXT_END, "")
                    text_item["text"] = f"{_STRUCTURE_TEXT_START}{text}{_STRUCTURE_TEXT_END}"
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

# ── veomni/data/transforms/wds_transform.py:100-117 ──
def _compact_structure_caption(value: str, sample_id: object) -> str:
    """``_compact_json_string`` 的调用侧包装：失败时带 sample id 记一条 error 再抛。

    只对「实际被 caption_pool 选中」的那一个 key 做，不是整个 caption_info 全做一遍 ——
    sft_* 那三个源一条样本带 4 个结构 key，全做等于在 IO worker 里白跑 3 次 json
    round-trip（单条 2k~7k token），而且某个没被选中的 key 坏了不该连带丢掉这条样本。
    """
    try:
        return _compact_json_string(value, mark_text_render=True)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        # 截断到 500 字符：完整 caption 有上万字符，刷日志没意义，靠 sample_id 回捞原文。
        logger.error(
            "Invalid Calico structure caption: sample_id=%r error=%s caption_prefix=%r",
            sample_id,
            exc,
            value[:500] if isinstance(value, str) else value,
        )
        raise

# ── veomni/data/transforms/wds_transform.py:120-134 ──
def _looks_like_structure_caption(value: object) -> bool:
    """这条 caption 是不是 calico 结构化 caption（整条 JSON，带 objects / text_render）。

    只给 ``_process_eval`` 用：评测 jsonl 没有源级别的 caption key 可以判别（统一叫
    ``caption``），所以靠内容嗅探，漏设 ``eval_prompt_format`` 时报错而不是静默按
    普通文本 BPE。判别条件取得紧（必须是 dict 且两个键都在），普通 caption 即使
    以 ``{`` 开头也不会误判。
    """
    if not isinstance(value, str) or not value.lstrip().startswith("{"):
        return False
    try:
        parsed = json.loads(value)
    except (ValueError, json.JSONDecodeError):
        return False
    return isinstance(parsed, dict) and "objects" in parsed and "text_render" in parsed

# ── veomni/data/transforms/wds_transform.py:137-151 ──
def _encode_structure_prompt(full_text: str, tokenizer) -> "torch.Tensor":
    """结构 caption 的 encode：哨兵内的 text_render 文字逐字符，其余照常 BPE。

    引号拆字的两个开关（split_quoted_text / quote_split_by_acc）不参与 —— 哨兵切分是
    这类数据的格式要求，不是可选的增强，见 ``encode_prompt`` 的 docstring。
    """
    return encode_prompt(
        full_text,
        tokenizer,
        None,
        "",
        "",
        quote_pairs=list(_STRUCTURE_QUOTE_PAIRS),
        decode_matched_json_strings=True,
    )

# ── veomni/data/transforms/wds_transform.py:154-160 ──
def prepare_extra_texts_new(extra_texts: list[dict]) -> tuple[list[str], np.ndarray]:
    txt_keys, probs = [], []
    for extra_text_item in extra_texts:
        key, prob = extra_text_item["key"], extra_text_item["prob"]
        txt_keys.append(key)
        probs.append(prob)
    return txt_keys, np.array(probs)

# ── veomni/data/transforms/wds_transform.py:163-165 ──
def pick_text_new(r: dict, txt_keys: list[str] = [], txt_probs: np.ndarray = np.array([])) -> str:
    txt, _ = pick_text_with_key(r, txt_keys, txt_probs)
    return txt

# ── veomni/data/transforms/wds_transform.py:168-188 ──
def pick_text_with_key(
    r: dict,
    txt_keys: list[str] = [],
    txt_probs: np.ndarray = np.array([]),
) -> tuple[str, str]:
    """Same selection as ``pick_text_new`` but also returns the chosen key
    (empty when nothing matched). Callers that need the key to infer the
    caption language (e.g., wds_und_transform's bilingual describe_prompt)
    use this; everyone else keeps ``pick_text_new``.
    """
    txt_infos = [(i, key) for i, key in enumerate(txt_keys) if r.get(key, None) is not None]
    if len(txt_infos) == 0:
        return "", ""

    txt_idx_, txt_keys_ = map(list, zip(*txt_infos))
    txt_probs_ = txt_probs[txt_idx_]
    txt_probs_ = txt_probs_ / txt_probs_.sum()
    txt_cumsum_ = np.cumsum(txt_probs_)

    idx = ((txt_cumsum_ - np.random.random()) < 0).sum()
    return r[txt_keys_[idx]], txt_keys_[idx]

# ── veomni/data/transforms/wds_transform.py:191-208 ──
def pick_text_with_fallback(r: dict, caption_pool: list[dict]) -> tuple[str, str]:
    if not any((e.get("fallback") for e in caption_pool)):
        txt_keys, txt_probs = prepare_extra_texts_new(caption_pool)
        return pick_text_with_key(r, txt_keys, txt_probs)
    infos: list[tuple[int, str]] = []
    for i, entry in enumerate(caption_pool):
        for key in [entry["key"], *(entry.get("fallback") or [])]:
            if r.get(key, None) is not None:
                infos.append((i, key))
                break
    if len(infos) == 0:
        return ("", "")
    idx_, keys_ = map(list, zip(*infos))
    probs_ = np.array([float(caption_pool[i]["prob"]) for i in idx_])
    probs_ = probs_ / probs_.sum()
    cumsum_ = np.cumsum(probs_)
    idx = (cumsum_ - np.random.random() < 0).sum()
    return (r[keys_[idx]], keys_[idx])

_LETTER = r"a-zA-ZÀ-ɏ̀-ͯͰ-ϿЀ-ӿ"

_APOSTROPHE = re.compile(rf"(?<=[{_LETTER}])['’](?=[{_LETTER}])" r"|['’](?=\d{2}s\b)")

_WORDCH = re.compile(rf"[{_LETTER}0-9]")

_AMBIGUOUS = frozenset("'\"")

_DEFAULT_QUOTE_PAIRS = (("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”"))

# ── veomni/data/transforms/wds_transform.py:224-233 ──
@functools.lru_cache(maxsize=None)
def _quote_tables(quote_pairs: tuple[tuple[str, str], ...]):
    """按引号配对表预计算：开引号集合、闭引号->可闭合的开引号集合、匹配所有引号字符的正则。"""
    openers = frozenset(q1 for q1, _ in quote_pairs)
    closers: dict[str, frozenset[str]] = {}
    for q1, q2 in quote_pairs:
        closers[q2] = closers.get(q2, frozenset()) | {q1}
    chars = "".join(sorted(openers | set(closers)))
    # 配对表为空时用一个永不匹配的正则，行为与逐字符扫描一致（直接得到空结果）
    return openers, closers, re.compile(f"[{re.escape(chars)}]" if chars else r"(?!)")

# ── veomni/data/transforms/wds_transform.py:236-333 ──
def _scan_quotation(
    prompt: str, quote_pairs: tuple[tuple[str, str], ...] | None = None, strict: bool = False
) -> list[tuple[int, int]] | None:
    """用栈匹配找出所有成对引号包住的区间，返回 [(开引号位置, 闭引号位置)]（闭区间，只留最外层）。
    引号不平衡时返回 None，表示这条文本的引号写法不可信。

    规则：
    - 引号按类型配对（直单引号配直单引号、直双引号配直双引号、‘ 配 ’、“ 配 ”），类似括号匹配；
    - 闭引号匹配栈内最近的同类开引号，中间未闭合的开引号视为杂散字符丢弃；
    - 只保留最外层区间，嵌套在内部的引号视为被引用文本的一部分，不单独切出；
    - 引号之间为空的区间（如 ""）不返回。

    直单引号与直双引号开闭同形，除引号外还常见于撇号（it's / players'）和英尺英寸号（5'9"），
    只靠栈配对会把它们当引号，实测会把两个所有格撇号配成一对、把中间整段描述当成图上文字，
    或者让整条文本判成不平衡。因此对这两个字符加位置判据（同印刷体智能引号的做法）：
    - 单词内部的撇号（it's / don’t）与年代缩写（’90s）直接跳过，不参与配对；
    - 只有落在「开引号位」（左边不是字母数字、右边不是空白）才允许作开引号；
    - 只有左边不是空白、且栈里有待闭合的同型开引号，才允许作闭引号；
    - 两者都不满足就当撇号 / 英寸号丢掉，既不进栈也不算错。
    弯引号 ‘ “ ” 没有这种歧义（只能作开或只能作闭），不施加位置判据，行为与括号匹配一致。

    判定为不平衡（返回 None）的只有两种情况，都只可能由弯引号引起：
    - 出现了找不到配对开引号的闭引号（且不在词尾撇号位置）；
    - 扫描结束时栈内仍有未闭合的 ‘ 或 “。
    收尾时栈里剩下的直引号（奇数个直双引号之类）当英寸号丢弃，不算错。

    实现上只在引号字符的位置上循环（正则一次找出所有引号位置），其余字符不逐个处理。

    ``strict=True`` 用于结构 caption 的哨兵对（\\ue000 / \\ue001）—— 这对定界符是程序自己
    插进去的、位置可控，不可能是撇号或英寸号，所以：
    - 不走撇号 / 位置判据（上面那套启发式是专为开闭同形的直引号写的）；
    - 多余的闭哨兵不再「当撇号丢掉」，一律判不平衡（返回 None），由调用方抛错；
    - 保留空区间（``"text": ""`` 产出的空哨兵对），让 split_quotation 把哨兵吃掉，
      否则哨兵会漏进普通片段、被当成正常字符 BPE 掉。
    """
    openers, closers, quote_char_pattern = _quote_tables(
        _DEFAULT_QUOTE_PAIRS if quote_pairs is None else tuple(tuple(p) for p in quote_pairs)
    )
    if not quote_char_pattern.search(prompt):
        return []
    # 不参与配对的撇号位置（文本里没有撇号类字符时省掉这一步；strict 模式用不上）
    skip_positions = (
        {m.start() for m in _APOSTROPHE.finditer(prompt)}
        if not strict and ("'" in prompt or "’" in prompt)
        else ()
    )
    stack: list[tuple[str, int]] = []  # (开引号字符, 位置)
    spans: list[tuple[int, int]] = []  # (起始位置, 结束位置)，闭区间
    n = len(prompt)
    for m in quote_char_pattern.finditer(prompt):
        i = m.start()
        if i in skip_positions:
            continue
        ch = m.group()
        prev = prompt[i - 1] if i else ""
        nxt = prompt[i + 1] if i + 1 < n else ""
        ambiguous = ch in _AMBIGUOUS

        # 1) 先尝试作为闭引号：从栈顶往下找最近的可配对开引号。直引号还要求左边不是空白
        matched = None
        if ch in closers and not (ambiguous and (prev == "" or prev.isspace())):
            can_close = closers[ch]
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] in can_close:
                    matched = j
                    break
        if matched is not None:
            spans.append((stack[matched][1], i))
            del stack[matched:]  # 弹出匹配项及其上方所有未闭合的开引号
            continue

        # 2) 再尝试作为开引号：弯引号无歧义直接进栈，直引号要求落在开引号位
        if ch in openers and (not ambiguous or (not _WORDCH.match(prev or " ") and nxt != "" and not nxt.isspace())):
            stack.append((ch, i))
            continue

        # 3) 既不能闭也不像开：直引号 / 词尾撇号当撇号丢掉，其余是真的多余闭引号。
        #    strict 模式不做这种挽救 —— 哨兵对不上就是上游打标坏了，必须让样本失败。
        if not strict and (ambiguous or _WORDCH.match(prev or " ")):
            continue
        return None

    if any(ch not in _AMBIGUOUS for ch, _ in stack):
        return None  # ‘ 或 “ 一直没被闭合
    # 剩下的未闭合直引号本来就歧义（英寸号 / 漏写），丢掉而不是判整条不可信

    # 只保留最外层区间：栈匹配得到的区间要么互不相交、要么完全嵌套，
    # 按起始位置排序后，落在上一个保留区间内部的就是内层区间，跳过
    spans.sort()
    result: list[tuple[int, int]] = []
    last_end = -1
    for start, end in spans:
        if start > last_end:
            # strict 模式保留空区间，否则那对哨兵会留在普通片段里（见 docstring）
            if strict or end > start + 1:  # 非 strict 下去掉 "" 这种空内容
                result.append((start, end))
            last_end = end
    return result

# ── veomni/data/transforms/wds_transform.py:336-394 ──
def split_quotation(
    prompt: str,
    quote_pairs: list[tuple[str, str]] | None = None,
    decode_matched_json_strings: bool = False,
) -> list[tuple[str, bool]]:
    """把 prompt 切成 (片段, 是否逐字符 tokenize) 的序列。

    普通 caption（``quote_pairs=None``）：按引号切，引号内的片段**带上引号本身**（与本仓
    历史的正则实现一致），片段拼接起来等于原文。下游 ``encode_prompt`` 对这些片段逐字符
    tokenize，其余片段照常走 BPE。引号不平衡时（``_scan_quotation`` 返回 None）当作这条
    文本里没有要渲染的文字：整条按普通文本返回 ``[(prompt, False)]``。漏切只是少一点字形
    监督，切错是直接给错标签。

    结构 caption：传 ``quote_pairs=[(_STRUCTURE_TEXT_START, _STRUCTURE_TEXT_END)]`` 和
    ``decode_matched_json_strings=True``。这时两点不一样 ——
    1) 哨兵本身不进结果，中间那段按 JSON 字符串解码一次再逐字符 tokenize。prompt 是
       ``json.dumps`` 出来的，``text_render[*].text`` 里的换行/引号在 prompt 里是
       ``\\n`` / ``\\"`` 这种两个字符的转义；不解码就会把反斜杠和 n 当成两个要渲染的
       字形喂进去（实测 infograph 每 100 条有 130+ 处这种转义）。因此这一支的片段
       拼接起来 **不等于** 原文，这是有意的。
    2) 哨兵不平衡不再退化成普通文本，直接抛 ValueError —— 结构 caption 整条就是
       prompt，哨兵是程序自己插的，对不上说明上游坏了，这条样本必须失败而不是静默
       降级成「整条 JSON 一个字形监督都没有」。

    >>> split_quotation('He said "hi" loudly.')
    [('He said ', False), ('"hi"', True), (' loudly.', False)]
    >>> split_quotation("the kittens' paws and the horses' coats")  # 所有格撇号不是引号
    [("the kittens' paws and the horses' coats", False)]
    """
    if not prompt:
        return []
    pairs = None if quote_pairs is None else tuple(tuple(p) for p in quote_pairs)
    spans = _scan_quotation(prompt, pairs, strict=decode_matched_json_strings)
    if spans is None:
        if decode_matched_json_strings:
            raise ValueError("Unbalanced structure text markers in prompt")
        return [(prompt, False)]

    actual_pairs = _DEFAULT_QUOTE_PAIRS if pairs is None else pairs
    result: list[tuple[str, bool]] = []
    pos = 0
    for start, end in spans:
        if start > pos:
            result.append((prompt[pos:start], False))
        matched_text = prompt[start : end + 1]
        if decode_matched_json_strings:
            quote_start, quote_end = prompt[start], prompt[end]
            if (quote_start, quote_end) not in actual_pairs:
                raise ValueError("Matched structure text markers do not match the configured pair")
            escaped_json_text = matched_text[len(quote_start) : len(matched_text) - len(quote_end)]
            # 借 json 自己的解码器处理 \\n / \\" / \\uXXXX，不手写反转义
            matched_text = json.loads(f'"{escaped_json_text}"')
        # 空内容不产出无意义的逐字符片段（strict 下空哨兵对就这样被吃掉）
        if matched_text:
            result.append((matched_text, True))
        pos = end + 1
    if pos < len(prompt):
        result.append((prompt[pos:], False))
    return result

# ── veomni/data/transforms/wds_transform.py:397-447 ──
def encode_prompt(
    prompt: str,
    tokenizer,
    text_tokenizer_max_length: int | None,
    prompt_template_encode_prefix: str,
    prompt_template_encode_suffix: str,
    split_quoted_text: bool = True,
    quote_splitter: Callable | None = None,
    quote_pairs: list[tuple[str, str]] | None = None,
    decode_matched_json_strings: bool = False,
) -> torch.Tensor:
    """把 prompt tokenize 成 input_ids；``split_quotation`` 标成 matched 的片段逐字符切。

    引号拆字的两个开关（源级别 / 逐条）：``split_quoted_text=False`` 整条按普通文本 BPE；
    ``quote_splitter`` 让调用方换一个切分函数，``Glm4ARImageTransform._resolve_quote_split``
    逐条按 ``caption_quotation_acc`` 决定时走它。

    ``quote_pairs`` / ``decode_matched_json_strings`` 原样转给 ``split_quotation``，结构
    caption 用它们把 text_render 的文字切成字形监督，见那边的 docstring。给了这两个就
    意味着「这条是结构 caption」—— 哨兵切分是数据本身的格式要求，**不受**上面那两个
    引号拆字开关影响（源级别把引号拆字关掉，结构化的字形监督照样要有）。
    """
    all_tokens: list[int] = []
    if quote_pairs is not None or decode_matched_json_strings:
        segments = split_quotation(prompt, quote_pairs, decode_matched_json_strings)
    elif split_quoted_text:
        segments = (quote_splitter or split_quotation)(prompt)
    else:
        segments = [(prompt, False)]
    for clean_prompt_sub, matched in segments:
        if matched:
            for sub_word in clean_prompt_sub:
                tokens = tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                all_tokens.extend(tokens)
        else:
            tokens = tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
            all_tokens.extend(tokens)

    if text_tokenizer_max_length is not None:
        all_tokens = all_tokens[:text_tokenizer_max_length]

    text_tokens_and_mask = tokenizer.pad(
        {"input_ids": [all_tokens]},
        max_length=len(all_tokens),
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = text_tokens_and_mask["input_ids"].squeeze(0)
    return input_ids

# ── veomni/data/transforms/wds_transform.py:450-495 ──
def resolve_quote_split(cfg, meta_info: dict, specs) -> tuple[bool, "Callable | None"]:
    """返回 ``(要不要拆字, 用哪个 splitter)``，喂给 ``encode_prompt``。

    自由函数而不是方法：t2i 血缘与 edit 血缘没有共同祖先，继承拿不到，两边各写一遍迟早漂。
    ``cfg`` 只要有 split_quoted_text / quote_split_by_acc / quote_acc_score_key /
    quote_split_when_unknown 四个属性。

    ``specs`` 是拼进**同一条**待编码文本的那些 caption，每项 ``(caption_key, idx)``；
    ``idx`` 非 None 表示这个 key 下是逐图 caption 列表、只看第 idx 条（acc 给标量时下标不起
    作用）。语义是 **AND**：有一条判为不可拆整条就不拆 —— 一条文本只编码一次，不存在"只拆
    其中一段"。
    """
    if not cfg.quote_split_by_acc:
        return (cfg.split_quoted_text, None)
    assert not cfg.split_quoted_text, (
        "quote_split_by_acc 与 split_quoted_text 互斥：前者逐条按 caption_quotation_acc 决定，后者对整源启用。"
    )
    score_info = meta_info.get("score_info")
    if isinstance(score_info, str):  # 有的源把 score_info 当 JSON 字符串存
        try:
            score_info = json.loads(score_info)
        except ValueError:
            score_info = None
    acc = score_info.get(cfg.quote_acc_score_key) if isinstance(score_info, dict) else None
    for caption_key, idx in specs:
        flag = acc.get(caption_key) if isinstance(acc, dict) else None
        listed = isinstance(flag, (list, tuple))
        if listed:
            flag = flag[idx] if idx is not None and -len(flag) <= idx < len(flag) else None
        if flag is None:
            why = (
                "这一行没有 score_info"
                if not isinstance(score_info, dict)
                else f"score_info 里没有 {cfg.quote_acc_score_key} 这个字段"
                if not isinstance(acc, dict)
                else f"{cfg.quote_acc_score_key}[{caption_key!r}] 是一串，但取不到第 {idx} 条"
                if listed
                else f"{cfg.quote_acc_score_key} 里没有这条 caption 的条目"
            )
            logger.warning_once(
                f"引号拆字查不到判据：{caption_key!r} —— {why}。按 quote_split_when_unknown={cfg.quote_split_when_unknown} 处理。⚠ 这条是 warning_once：每个 dataloader worker 每个 caption key 只打一次，而且只有 LOCAL_RANK=0 的进程打 ⇒ 它只说明「出现过」，**读不出缺失率**；要缺失率得离线扫 parquet。"
            )
            flag = cfg.quote_split_when_unknown
        if not flag:
            return (False, split_quotation)  # AND：有一条判不可拆，整条就不拆
    return (True, split_quotation)

# ── veomni/data/transforms/wds_transform.py:498-506 ──
def flatten_nested_dict(d: dict) -> dict:
    """Flatten nested dicts into a single dict. with key xx.yy"""
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update({f"{k}.{k2}": v2 for k2, v2 in v.items()})
        else:
            flat_dict[k] = v
    return flat_dict

# ── veomni/data/transforms/wds_transform.py:509-535 ──
def build_image_token_type_ids(
    latent_shape: torch.Tensor,
    st_idx: int = 0,
    gen_enabled: bool = False,
) -> torch.Tensor:
    grid_t, grid_h, grid_w = latent_shape
    type_id = 2 if gen_enabled else 1

    t_index = torch.arange(st_idx + 1, st_idx + 1 + grid_t).reshape(-1, 1).repeat(1, grid_h * grid_w).flatten()
    h_index = (
        torch.arange(st_idx + 1, st_idx + 1 + grid_h)
        .reshape(1, -1, 1)
        .repeat(1, 1, grid_w)
        .repeat(grid_t, 1, 1)
        .flatten()
    )
    w_index = (
        torch.arange(st_idx + 1, st_idx + 1 + grid_w)
        .reshape(1, 1, -1)
        .repeat(1, grid_h, 1)
        .repeat(grid_t, 1, 1)
        .flatten()
    )
    image_token_type_ids = torch.stack([t_index, h_index, w_index, torch.ones_like(t_index) * type_id], dim=1)
    start_token_type = torch.tensor([st_idx] * 3 + [type_id])[None, :]
    end_token_type = torch.tensor([st_idx + 1 + latent_shape.max()] * 3 + [1])[None, :]
    return torch.cat([start_token_type, image_token_type_ids, end_token_type], dim=0)

# ── veomni/data/transforms/wds_transform.py:538-549 ──
def build_text_token_type_ids(
    length: int,
    st_idx: int = 0,
) -> torch.Tensor:
    text_token_type_ids = torch.cat(
        [
            torch.arange(st_idx, st_idx + length).reshape(-1, 1).repeat(1, 3),
            torch.zeros((length, 1), dtype=torch.int32),
        ],
        dim=1,
    )
    return text_token_type_ids

# ── veomni/data/transforms/wds_transform.py:984-1526 ──
@dataclass
class Glm4ARImageTransform:
    """
    Transform for online VQ: processes raw images from WebDataset.

    Sequence layout (FFR-aligned, single image):
      default (shape_info given, loss=0):
        ``<|user|> <|gen_image|> {caption} <|assistant|> <sop>H W<eop>
          <boi> {ph}*N <eoi> <|user|>``
      shape_info_trainable=True (model emits shape_info, loss=1):
        ``<|user|> <|gen_image|> {caption} <|assistant|> <sop>H W<eop>
          <boi> {ph}*N <eoi> <|user|>`` (same layout, only loss mask differs)

    ``<|gen_image|>`` (= ``gen_image_request_token_id``, default 154857) is
    a single untrained-slot token placed at the START of the user turn
    that signals "the following caption is an image-generation request" —
    language-agnostic replacement for the previous ~5-token text prompt
    "Generate an image: ". Registered in the tokenizer (added_tokens) so
    it round-trips through decode for log inspection.

    Trailing ``<|user|>`` is the FFR conversation-end marker (loss=1 at the
    predict-trailing-user position so the model learns to emit EOT).
    Loss target spans [<|assistant|>, image span, trailing <|user|>] in
    both modes; when ``shape_info_trainable`` is on, [<sop>H W<eop>] is
    also loss=1.

    ``model.forward`` resolves ``labels == gen_placeholder_token`` and
    substitutes ``vq_indices + lang_vocab_size``. Outputs the same
    input_ids / token_type_ids / loss_masks schema as Glm4ARImageUndTransform.
    """

    image_keys: list[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
    )
    vae_latent_scale: int = 16
    scale_factor: int = 16
    image_max_pixels: int = 1536 * 1536
    image_min_pixels: int = 256 * 256
    # Hard reject filter applied BEFORE _smart_resize. None disables.
    image_min_pixels_filter: int | None = None
    max_ratio: float | None = 4.0
    vit_patch_size: int = 16

    # Compression-ratio aligned single-side resize: VQ-ViT input is resized so
    # its output grid matches the VAE latent grid. Defaults preserve old
    # behavior (no resize) when both are equal.
    vq_compression_ratio: int = 16
    vae_compression_ratio: int = 16

    # VQ projector pre-pool factor — mirror of the ckpt's
    # ``vq_vision_config.vq_pre_pool_factor``. The ViT consumes the ORIGINAL
    # image (raw grid = pixels // vit_patch_size, one pixel_values row per
    # cell), but the projector pools by ``pf`` afterwards, so the AR vision
    # span (placeholders / position / shape_info) uses the POST-pool grid
    #   vision_grid = raw_grid // pf      (effective compression = patch * pf)
    # and ``vit_image_grid_thw`` is EMITTED as this post-pool LLM grid (the
    # unified train_ar contract — vq_encode upscales ×pf internally to recover
    # the raw patch grid matching pixel_values rows).
    # Set ``scale_factor = vit_patch_size * pf`` so resized sides divide evenly.
    vq_pre_pool_factor: int = 1

    # Optional per-sample low-resolution bias. With probability ``prob`` the
    # per-sample max-pixel cap is overridden to a value drawn log-uniformly
    # from ``[min_pixels, max_pixels]`` (forces _smart_resize to downscale
    # more aggressively). Mirrors ``Glm4ARImageUndTransform.low_res_bias`` so
    # the same yaml stanza works for both und and gen sources.
    # Schema: ``{"prob": float, "min_pixels": int, "max_pixels": int}``.
    low_res_bias: dict | None = None

    # ── Resize mode (ported from cogview decoder transform) ──
    # "smart_resize" (default): the existing _smart_resize path — clamp area
    #     into [image_min_pixels, image_max_pixels] aligned to scale_factor.
    # "maintain_aspect_ratio": pick a random target side ∈ [image_resize_min_length,
    #     image_resize_max_length] (or Gaussian around `preferred_size` with
    #     probability `mix_ratio`), then scale-down keeping aspect, and crop
    #     each side to a multiple of `edge_padding_size`.
    image_resize_min_length: int | None = 256
    image_resize_max_length: int | None = 256
    edge_padding_size: int = 32

    # boi/eoi/placeholder IDs in input_ids. This transform is the *online-VQ encode (pixel)*
    # path so gen_placeholder_token = config.image_token_id = 154856.
    gen_boi_token: int = 154830
    gen_eoi_token: int = 154831
    gen_placeholder_token: int = 154856

    # Chat-template role IDs (unified tokenizer). FFR-aligned layout uses
    # ``<|user|>`` both as turn opener and as conversation-end terminator.
    user_token_id: int = USER_TOKEN_ID
    assistant_token_id: int = ASSISTANT_TOKEN_ID
    # Single-token gen-request marker (default 154857), placed at the START
    # of the user turn (before the caption). The token is also registered
    # in the tokenizer (added_tokens) as ``<|gen_image|>`` so it round-trips
    # through decode for log dumps.
    gen_image_request_token_id: int = GEN_IMAGE_REQUEST_TOKEN_ID
    # When True, shape_info moves into loss=1 so the model learns
    # to emit its own (H, W) prediction.
    shape_info_trainable: bool = False

    vision_token_dropout_rate: float = 0.0
    vae_pixel_dropout_rate: float = 0.0
    # CFG-style condition dropout. The "text part" is EVERYTHING outside the
    # VQ image span (role markers, <|gen_image|>, caption, shape_info,
    # trailing <|user|>) — exactly complementary to keep_vision_token. Both
    # modes set AR loss to all-0 (labels all -100); the AR sequence serves
    # purely as the diffusion branch's prefix condition:
    #   text_dropout_rate        — drop the text part; sequence = <boi> ph*N
    #                              <eoi> (diff conditioned on VQ tokens only)
    #   text_vision_dropout_rate — drop text AND VQ span; sequence = the
    #                              minimal <|user|> marker pair (uncond diff;
    #                              tokenizer("") is 0 tokens so a fully-empty
    #                              segment is not representable)
    # All four rates share one Bernoulli draw → mutually exclusive; their
    # sum must be ≤ 1.
    text_dropout_rate: float = 0.0
    text_vision_dropout_rate: float = 0.0

    tokenizer: AutoTokenizer | None = None
    caption_pool: list[dict] = field(default_factory=list)

    split_quoted_text: bool = True
    quote_split_by_acc: bool = False
    quote_acc_score_key: str = "caption_quotation_acc"
    quote_split_when_unknown: bool = False

    # 按 meta.jsonl 的字段阈值过滤样本（源级别，默认不开）。详见
    # veomni/data/transforms/meta_filter.py 的模块注释。典型写法：
    #   meta_filters:
    #   - {key: portrait_p_keep, op: '>', value: 0.4, on_missing: drop}
    # ⚠ 过滤发生在采样之后，不会自动补 weight —— 通过率低就等于把这一源的
    #   实际占比按通过率等比例缩了。
    meta_filters: list[dict] | None = None
    meta_filter_on_missing: str = "keep"

    def __post_init__(self):
        # 子类若自己定义 __post_init__，记得 super().__post_init__()，否则判据不校验。
        self._meta_filter_rules = compile_meta_filters(self.meta_filters, self.meta_filter_on_missing)

    def _check_meta_filters(self, meta_info: dict) -> None:
        # dataclass 不是每条路径都走 __post_init__（有的地方直接 copy/replace 实例），
        # 取不到编译结果时就地补一次。
        rules = getattr(self, "_meta_filter_rules", None)
        if rules is None:
            rules = self._meta_filter_rules = compile_meta_filters(self.meta_filters, self.meta_filter_on_missing)
        check_meta_filters(meta_info, rules)

    def _resolve_quote_split(self, meta_info: dict, caption_key: str) -> tuple[bool, "Callable | None"]:
        """t2i 这一条文本只有一个 caption ⇒ 单 spec 的 ``resolve_quote_split``（见那里）。"""
        return resolve_quote_split(self, meta_info, [(caption_key, None)])


    # ---- image processing helpers ----

    @staticmethod
    def _open_image(image: bytes | dict | Image.Image | str) -> Image.Image:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, dict):
            image = Image.open(io.BytesIO(image["bytes"])) if "bytes" in image else Image.open(image["path"])
        elif isinstance(image, str):
            image = Image.open(image)
        return image.convert("RGB")

    def _maintain_aspect_random_side(self, image: Image.Image) -> int:
        """Pick a random target side in [min, max] (or Gaussian around
        ``preferred_size`` with probability ``mix_ratio``), then run
        maintain-aspect-ratio resize-and-crop."""
        min_side = self.image_resize_min_length
        max_side = self.image_resize_max_length
        assert min_side is not None and max_side is not None, (
            "resize_mode='maintain_aspect_ratio' requires image_resize_min_length and image_resize_max_length"
        )
        assert min_side <= max_side, f"image_resize_min_length={min_side} must be ≤ image_resize_max_length={max_side}"
        if min_side == max_side:
            target_side = min_side
        else:
            target_side = random.randint(min_side, max_side)
        return target_side

    def _sample_max_pixels(self) -> int:
        """Draw the per-sample max-pixel cap. Falls back to image_max_pixels
        when low_res_bias is disabled or the Bernoulli gate misses."""
        if not self.low_res_bias:
            return self.image_max_pixels
        prob = float(self.low_res_bias.get("prob", 0.0))
        if random.random() >= prob:
            return self.image_max_pixels
        lo = float(self.low_res_bias["min_pixels"])
        hi = float(self.low_res_bias["max_pixels"])
        return int(math.exp(random.uniform(math.log(lo), math.log(hi))))

    def _smart_resize(self, image: Image.Image, max_pixels: int | None = None) -> Image.Image:
        width, height = image.size
        sf = self.scale_factor
        h_bar = max(sf, round(height / sf) * sf)
        w_bar = max(sf, round(width / sf) * sf)
        # random_resize_side = self._maintain_aspect_random_side(image)
        per_sample_max = self.image_max_pixels if max_pixels is None else max_pixels
        # target_side = min(random_resize_side, per_sample_max ** 0.5)
        # max_px = int(target_side ** 2)

        max_px = per_sample_max

        # Keep the floor consistent: never upsample past the per-sample cap.

        if h_bar * w_bar > max_px:
            beta = math.sqrt((height * width) / max_px)
            h_bar = math.floor(height / beta / sf) * sf
            w_bar = math.floor(width / beta / sf) * sf

        return image.resize((w_bar, h_bar), resample=Image.Resampling.LANCZOS)

    def _patchify(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a PIL image to a patchified pixel tensor for the vision encoder.

        Returns:
            vit_pixel_values: (num_patches, patch_dim) tensor
            vit_grid_thw: (3,) tensor [t, h_patches, w_patches]
        """
        c, h, w = image_tensor.shape
        ph = pw = self.vit_patch_size
        grid_h, grid_w = h // ph, w // pw
        vit_pixel_values = rearrange(image_tensor, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=ph, pw=pw)
        vit_grid_thw = torch.tensor([1, grid_h, grid_w])
        return vit_pixel_values, vit_grid_thw

    # ---- main __call__ ----

    def __call__(self, item: dict) -> dict | None:
        if self.tokenizer is None:
            raise ValueError("Glm4ARImageTransform.tokenizer 未设置，请在外部注入 tokenizer 实例。")

        tokenizer = self.tokenizer
        vae_latent_scale = self.vae_latent_scale

        # -- parse meta --
        if isinstance(item.get("meta_info"), str):
            meta_info = json.loads(item["meta_info"])
        else:
            meta_info = item.get("meta_info", {})

        # -- meta 数值过滤（放在解图之前，判掉的样本不白解一次 JPEG）--
        self._check_meta_filters(meta_info)

        # -- caption --
        if self.caption_pool:
            txt_keys, txt_probs = prepare_extra_texts_new(self.caption_pool)
            txt = pick_text_new(meta_info, txt_keys, txt_probs)
        else:
            txt = ""

        if not txt:
            caption_keys = [
                "long_caption_zh",
                "long_caption_en",
                "mid_caption_zh",
                "mid_caption_en",
                "short_caption_zh",
                "short_caption_en",
            ]
            caption_keys = [k for k in caption_keys if meta_info.get(k) is not None]
            if not caption_keys:
                raise ValueError(f"no caption key found in {item.get('__key__', '?')} {meta_info}")
            txt = meta_info[random.choice(caption_keys)]

        # -- open image --
        img_bytes = None
        for image_key in self.image_keys:
            if image_key in item and item[image_key] is not None:
                img_bytes = item[image_key]
                break
        if img_bytes is None:
            raise ValueError(f"no image bytes found in {item.get('__key__', '?')}")

        image = self._open_image(img_bytes)

        # -- hard min-pixel filter (raw image, before resize) --
        if self.image_min_pixels_filter is not None:
            w, h = image.size
            if w * h < self.image_min_pixels_filter:
                raise ValueError(f"image area {w * h} < image_min_pixels_filter {self.image_min_pixels_filter}")

        # -- aspect ratio filter --
        if self.max_ratio is not None:
            w, h = image.size
            ratio = max(w, h) / max(min(w, h), 1)
            if ratio > self.max_ratio:
                raise ValueError(f"aspect ratio {ratio:.1f} > max_ratio {self.max_ratio}")

        image = self._smart_resize(image, max_pixels=self._sample_max_pixels())
        image_tensor = to_tensor(image)
        image_tensor = normalize(image_tensor, 0.5, 0.5)
        vae_pixel_values = image_tensor.unsqueeze(0)  # bchw, full resolution for VAE

        vit_image_tensor = image_tensor
        # Compression-ratio aligned resize for VQ-ViT input so its output grid
        # matches the VAE latent grid. Done here (pre-patchify) instead of in
        # the train loop, since vit_pixel_values leaves this transform already
        # flattened to (N_patches, C*ph*pw) and can no longer go through
        # F.interpolate.
        # assert self.vq_compression_ratio % self.vit_patch_size == 0
        # assert self.vae_compression_ratio % self.vae_latent_scale == 0
        # if self.vq_compression_ratio != self.vae_compression_ratio:
        #     scale = self.vq_compression_ratio / self.vae_compression_ratio
        #     vit_image_tensor = F.interpolate(
        #         vae_pixel_values,
        #         scale_factor=scale,
        #         mode="bicubic",
        #         align_corners=False,
        #         antialias=True,
        #     ).squeeze(0)
        # else:
        #     vit_image_tensor = image_tensor
        vit_pixel_values, vit_image_grid_thw = self._patchify(vit_image_tensor)

        # -- AR-side vision-token grid: POST-pre_pool ----------------------
        # The projector pools the ViT output by ``vq_pre_pool_factor``, so the
        # number of vision tokens the AR actually receives is raw // pf per
        # side (effective compression = vit_patch_size * pf, e.g. 16*2=32).
        # The local ``vit_image_grid_thw`` below is the RAW patchify grid
        # (internal only); the EMITTED ``vit_image_grid_thw`` field is the
        # post-pool ``vit_image_grid_thw`` (unified train_ar contract).
        pf = self.vq_pre_pool_factor
        assert int(vit_image_grid_thw[1]) % pf == 0 and int(vit_image_grid_thw[2]) % pf == 0, (
            f"vit grid {vit_image_grid_thw.tolist()} not divisible by vq_pre_pool_factor={pf}; "
            f"set scale_factor={self.vit_patch_size * pf} so resized sides align."
        )
        vit_image_grid_thw = vit_image_grid_thw.clone()
        vit_image_grid_thw[1] = vit_image_grid_thw[1] // pf
        vit_image_grid_thw[2] = vit_image_grid_thw[2] // pf

        # -- compute **patchified** latent shape (for token_type_ids / position encoding) --
        vae_latent_h = image.height // vae_latent_scale
        vae_latent_w = image.width // vae_latent_scale
        vae_latent_shape = torch.tensor([1, vae_latent_h, vae_latent_w])

        # -- modality dropout decision (mutually-exclusive Bernoulli split) --
        # keep_vision_token controls the discrete AR VQ branch (image span +
        # vit_pixel_values); keep_vae_pixel controls the diffusion branch;
        # keep_text controls the caption span (CFG-style condition dropout —
        # layout markers and shape_info stay, only the caption empties).
        _p_vis = self.vision_token_dropout_rate
        _p_vae = self.vae_pixel_dropout_rate
        _p_txt = self.text_dropout_rate
        _p_txt_vis = self.text_vision_dropout_rate
        assert _p_vis + _p_vae + _p_txt + _p_txt_vis <= 1.0 + 1e-6, (
            "vision_token_dropout_rate + vae_pixel_dropout_rate + "
            "text_dropout_rate + text_vision_dropout_rate must be ≤ 1 "
            f"(got {_p_vis} + {_p_vae} + {_p_txt} + {_p_txt_vis})"
        )
        keep_vision_token, keep_vae_pixel, keep_text = True, True, True
        if _p_vis + _p_vae + _p_txt + _p_txt_vis > 0.0:
            r = random.random()
            if r < _p_vis:
                keep_vision_token = False  # diff-only: drop AR VQ span
            elif r < _p_vis + _p_vae:
                keep_vae_pixel = False  # AR-only: drop diffusion target
            elif r < _p_vis + _p_vae + _p_txt:
                keep_text = False  # VQ-only condition: drop all text tokens
            elif r < _p_vis + _p_vae + _p_txt + _p_txt_vis:
                keep_text = False  # uncond: drop text AND VQ span
                keep_vision_token = False  # → minimal <|user|> pair

        # -- FFR-aligned role-wrapped layout ------------------------------
        # Layout (fixed; shape_info_trainable only flips its loss bits):
        #   <|user|> <|gen_image|> {caption} <|assistant|> <sop>H W<eop>
        #   <boi> {ph}*N <eoi> <|user|>
        # The single-token <|gen_image|> sits at the START of the user
        # turn, right after the user marker — language-agnostic "produce
        # image" signal that replaces the previous text "Generate an
        # image: " prompt.
        # Loss=1 by default on [<|assistant|>, <boi>, ph*N, <eoi>, trailing
        # <|user|>]; the shape_info span is loss=0 unless
        # ``shape_info_trainable=True``, in which case it joins loss=1.
        caption_tokens = encode_prompt(txt, tokenizer, None, "", "")
        shape_info_txt = f"<sop>{vit_image_grid_thw[1]} {vit_image_grid_thw[2]}<eop>"
        shape_info_tokens = encode_prompt(shape_info_txt, tokenizer, None, "", "")

        user_tok = torch.tensor([self.user_token_id], dtype=torch.int32)
        gen_tok = torch.tensor([self.gen_image_request_token_id], dtype=torch.int32)
        asst_tok = torch.tensor([self.assistant_token_id], dtype=torch.int32)
        tail_user_tok = user_tok.clone()  # FFR-style conversation end marker

        img_tokens = torch.cat(
            [
                torch.tensor([self.gen_boi_token], dtype=torch.int32),
                torch.full((vit_image_grid_thw.prod().item(),), self.gen_placeholder_token, dtype=torch.int32),
                torch.tensor([self.gen_eoi_token], dtype=torch.int32),
            ]
        )

        n_cap = caption_tokens.shape[0]
        n_si = shape_info_tokens.shape[0]
        n_img = img_tokens.shape[0]

        # -- token_type_ids: thread st_idx through each span --------------
        # The VQ image span (img_tokens / img_tt) is present only when the
        # vision-token branch is kept; for the diff-only (vision_token_dropout)
        # path the image is generated by diffusion alone, so the AR sequence is
        # text-only and the trailing <|user|> directly follows shape-info.
        if keep_text:
            st = 0
            user_tt = build_text_token_type_ids(1, st_idx=st)
            st += 1
            gen_tt = build_text_token_type_ids(1, st_idx=st)
            st += 1
            cap_tt = build_text_token_type_ids(n_cap, st_idx=st)
            st += n_cap
            asst_tt = build_text_token_type_ids(1, st_idx=st)
            st += 1
            si_tt = build_text_token_type_ids(n_si, st_idx=st)
            st += n_si
            # diff position anchor = AR image-span anchor (st_idx + 1). build_image_token_type_ids
            # places the image's t/h/w at st_idx+1, so the diffusion branch uses the SAME
            # anchor to align with the vision tokens. Even for the diff-only path
            # (vision_token_dropout, where the discrete VQ span is NOT materialized in the AR
            # sequence) the generated image still conceptually occupies the slot right after
            # shape-info, so its diff positions start at st+1 too — keeping them contiguous
            # after the text condition (vs overlapping it if anchored at 0).
            diff_pos_anchor = st + 1
            if keep_vision_token:
                img_tt = build_image_token_type_ids(vit_image_grid_thw, st_idx=st, gen_enabled=True)
                st += int(vit_image_grid_thw.max().item()) + 2
            tail_tt = build_text_token_type_ids(1, st_idx=st)

            # -- concat tokens & token_type_ids (image span optional) -----
            token_segs = [user_tok, gen_tok, caption_tokens, asst_tok, shape_info_tokens]
            tt_segs = [user_tt, gen_tt, cap_tt, asst_tt, si_tt]
            if keep_vision_token:
                token_segs.append(img_tokens)
                tt_segs.append(img_tt)
            token_segs.append(tail_user_tok)
            tt_segs.append(tail_tt)
            tokens = torch.cat(token_segs, dim=0)
            token_type_ids = torch.cat(tt_segs, dim=0)

            # -- loss_masks -----------------------------------------------
            asst_pos = 1 + 1 + n_cap  # user + <|gen_image|> + caption
            si_start = asst_pos + 1

            loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)
            if keep_vision_token:
                # <|assistant|> + image span + trailing <|user|> loss=1。trailing
                # <|user|> 教模型从 <eoi> emit FFR-style EOT;shape span(可选)教分辨率。
                loss_masks[asst_pos] = 1.0
                img_start = si_start + n_si
                img_end = img_start + n_img  # exclusive (= trailing user position)
                loss_masks[img_start : img_end + 1] = 1.0
                if self.shape_info_trainable:
                    loss_masks[si_start : si_start + n_si] = 1.0
            # else: diff-only(vision_token_dropout)样本 → AR loss 全 0,只作 diffusion 的
            # prefix condition、不训练 AR(对称于 vae_pixel_dropout → diff loss=0)。原来此处
            # 标 si→trailing<|user|> 的 EOT,会与正常样本的 si→<boi> 冲突,并用低 loss 的
            # asst/EOT 稀释 vision-token 预测,故 diff-only 样本不再计 AR loss。
        else:
            # -- complementary text drop (CFG uncond) ----------------------
            # keep_text governs EVERYTHING outside the VQ image span (role
            # markers, <|gen_image|>, caption, shape_info, trailing <|user|>)
            # — exactly complementary to keep_vision_token. The AR side never
            # trains on these samples (loss all-0 → labels all -100); the
            # sequence exists purely as the diffusion branch's prefix
            # condition KV.
            if keep_vision_token:
                # uncond AR span: only <boi> ph*N <eoi>. The diff branch is
                # conditioned on the image's own discrete VQ tokens with NO
                # text — matching the inference-time uncond pass that keeps
                # already-generated VQ tokens but drops the text prompt.
                tokens = img_tokens
                token_type_ids = build_image_token_type_ids(vit_image_grid_thw, st_idx=0, gen_enabled=True)
                diff_pos_anchor = 1  # t/h/w start at st_idx+1 = 1
            else:
                # text AND vision dropped: a fully-empty AR segment is not
                # representable (tokenizer("") == 0 tokens; after the FFR
                # pre-shift input_ids would be empty → 0-length packed
                # segment), so keep the minimal <|user|> marker pair. The
                # diff branch sees a near-empty prefix — the unconditional
                # condition.
                tokens = torch.cat([user_tok, tail_user_tok], dim=0)
                token_type_ids = torch.cat(
                    [
                        build_text_token_type_ids(1, st_idx=0),
                        build_text_token_type_ids(1, st_idx=1),
                    ],
                    dim=0,
                )
                diff_pos_anchor = 2  # conceptual image slot after <|user|>
            loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)

        assert tokens.shape[0] == token_type_ids.shape[0]

        # -- labels: pre-shifted (FFR-style) ------------------------------
        # After shift, label slots at ``ph`` natively carry ``gen_placeholder_token``;
        # ``model.forward`` finds them via ``labels == placeholder_token_id`` and
        # substitutes ``vq_indices + lang_vocab_size`` from vq_visual.
        labels = tokens[1:].clone().to(torch.long)
        shifted_loss_mask = loss_masks[1:]
        labels[shifted_loss_mask == 0] = -100
        loss_masks = shifted_loss_mask

        input_ids = tokens[:-1].to(torch.long)
        token_type_ids = token_type_ids[:-1]

        out_item = {
            # All per-token fields share the SAME length N-1 (the pre-shifted
            # FFR convention): input_ids = tokens[:-1] pairs with labels =
            # tokens[1:]. Emitting the full-length ``tokens`` here instead
            # desyncs input_ids (N) from position_ids/labels/cu_seqlens (N-1)
            # and blows up at RoPE with `q(ΣN) * cos(Σ(N-1))`.
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids.to(torch.long),
            "loss_masks": loss_masks,
            "seq_lens": torch.tensor([input_ids.shape[0]]),
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "use_quantize": [True],
            "placeholder_token_id": torch.tensor(self.gen_placeholder_token, dtype=torch.long),
            "has_diff": keep_vae_pixel,
            "source_name": item.get("source_name", ""),
            "ds_idx": item.get("ds_idx", 0),
        }

        # Unified train_ar contract: ``vit_image_grid_thw`` carries the
        # LLM-token (post-pre_pool) grid. One field, three consumers:
        #   • sizes the AR placeholder span (placeholder count == prod()),
        #   • feeds vq_encode_to_codebook, which upscales ×pre_pool_factor
        #     internally to recover the raw patch grid (pixel_values rows),
        #   • is the position scale the diff branch interpolates onto
        #     (diff pos = anchor + i × vision/vae).
        # Emitted unconditionally: the collator's vision branch is gated on
        # vit_pixel_values presence, its diff branch on has_diff — both read
        # this field, so diff-only (vision-dropped) samples still carry it.
        out_item["vit_image_grid_thw"] = [vit_image_grid_thw.to(torch.long)]

        if keep_vision_token:
            out_item["vit_pixel_values"] = [vit_pixel_values]

        if keep_vae_pixel:
            out_item["vae_pixel_values"] = [vae_pixel_values]
            out_item["vae_latent_shape"] = [vae_latent_shape]
            out_item["diff_pos_anchor"] = [torch.tensor(diff_pos_anchor, dtype=torch.long)]

        return out_item

# ── veomni/data/transforms/wds_transform.py:1529-2015 ──
@dataclass
class Glm4TransfusionImageTransform(Glm4ARImageTransform):
    """Gen transform for the Transfusion model (single tower, shared params)."""

    get_rope_index: Callable | None = None
    edit_with_t2i_caption_ratio: float = 0.5
    template_mode: str = "sft_mode"
    vae_token_id: int | None = None
    # 源级别的推理 prompt 格式，只有 _process_eval 读它。设成 "calico_structure" 时
    # 推理侧按训练侧同一套处理结构 caption（压缩 + 删 type + text_render 逐字符），
    # 否则评测喂的是带 type、带 JSON 转义的原始 JSON，和训练分布对不上。
    # 非 None 还会让 __call__ 直接走 eval 分支（评测数据行不一定带 mode 字段）。
    eval_prompt_format: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.template_mode not in ("sft_mode", "pretrain_mode"):
            raise ValueError("template_mode must be 'sft_mode' or 'pretrain_mode'")
        if self.eval_prompt_format not in (None, "calico_structure"):
            raise ValueError(
                f"eval_prompt_format 目前只支持 None / 'calico_structure'，得到 {self.eval_prompt_format!r}"
            )

    def mask_to_block_range(self, mask: torch.BoolTensor) -> torch.IntTensor:
        mask = mask
        starts = mask & ~F.pad(mask[:-1], (1, 0))
        ends = mask & ~F.pad(mask[1:], (0, 1))
        start_idx = starts.nonzero(as_tuple=True)[0]
        end_idx = ends.nonzero(as_tuple=True)[0] + 1
        block_range = torch.stack([start_idx, end_idx], dim=-1)
        return block_range

    def __call__(self, item: dict) -> dict | None:
        if item.get("mode", "train") == "eval" or self.eval_prompt_format is not None:
            return self._process_eval(item)
        elif item.get("relation_info_binary") is not None:
            return self._process_edit(item)
        elif any(item.get(key) is not None for key in ["file_path_binary", *self.image_keys]):
            return self._process_t2i(item)
        else:
            raise ValueError(f"unknown item type {item.keys()}")

    def _process_t2i(self, item: dict) -> dict:
        if self.tokenizer is None:
            raise ValueError("Glm4TransfusionImageTransform.tokenizer 未设置")
        tokenizer = self.tokenizer
        vae_latent_scale = self.vae_latent_scale

        meta = item.get("meta_info", item.get("caption_info", {}))
        meta = json.loads(meta) if isinstance(meta, str) else meta
        if "meta_info" not in item:
            meta = flatten_nested_dict(meta)
        # meta 数值过滤：放在解图之前。⚠ 走 caption_info 的 parquet 源这里已经拍平过，
        # 判据 key 要按拍平后的名字写；wds 的 meta_info 是原始嵌套，用 a.b 下钻。
        self._check_meta_filters(meta)
        caption, caption_key = pick_text_with_fallback(meta, self.caption_pool) if self.caption_pool else ("", "")
        if not caption:
            keys = [key for key in (
                "long_caption_zh", "long_caption_en", "mid_caption_zh", "mid_caption_en",
                "short_caption_zh", "short_caption_en",
            ) if meta.get(key) is not None]
            if not keys:
                raise ValueError(f"no caption key found in {item.get('__key__', '?')}")
            caption_key = random.choice(keys)
            caption = meta[caption_key]
        split_quoted, quote_splitter = self._resolve_quote_split(meta, caption_key)

        # -- open + filter + resize image (same as ARDiff) --
        img_bytes = next((item[key] for key in ["file_path_binary", *self.image_keys] if item.get(key) is not None), None)
        if img_bytes is None:
            raise ValueError(f"no image bytes found in {item.get('__key__', '?')}")
        image = self._open_image(img_bytes)
        if self.image_min_pixels_filter is not None:
            w, h = image.size
            if w * h < self.image_min_pixels_filter:
                raise ValueError(f"image area {w * h} < image_min_pixels_filter {self.image_min_pixels_filter}")
        if self.max_ratio is not None:
            w, h = image.size
            if max(w, h) / max(min(w, h), 1) > self.max_ratio:
                raise ValueError(f"aspect ratio > max_ratio {self.max_ratio}")
        image = self._smart_resize(image, max_pixels=self._sample_max_pixels())
        image_tensor = normalize(to_tensor(image), 0.5, 0.5)
        vae_pixel_values = image_tensor.unsqueeze(0)  # bchw, full-res for VAE

        # -- VAE-latent grid drives both the shape token and diff positions --
        vae_latent_h = image.height // vae_latent_scale
        vae_latent_w = image.width // vae_latent_scale
        vae_grid_thw = torch.tensor([[1, vae_latent_h, vae_latent_w]])

        # CFG condition dropout
        keep_text = not (self.text_dropout_rate > 0.0 and random.random() < self.text_dropout_rate)
        caption = caption if keep_text else ""

        # Calico structure caption：整条 JSON 当 prompt 用，先压成单行、删掉 type、把
        # text_render[*].text 用哨兵包起来，下面 encode 时哨兵内的文字逐字符 tokenize。
        # 放在 dropout 之后：被 drop 掉的样本 caption 已是空串，不用白跑这趟 json。
        is_structure_caption = bool(caption) and caption_key in _CALICO_STRUCTURE_CAPTION_KEYS
        if is_structure_caption:
            caption = _compact_structure_caption(caption, item.get("uuid") or item.get("__key__") or "unknown")

        if self.template_mode == "pretrain_mode":
            # The registered BOI/EOI tokens delimit BPE exactly as in the
            # previous segmented encoder. Keep image placeholders as IDs.
            shape_ids = tokenizer.encode(f"{image.height} {image.width}", add_special_tokens=False)
            if keep_text and hasattr(tokenizer, "convert_ids_to_tokens"):
                if not hasattr(self, "_boundary_strings"):
                    boi = tokenizer.convert_ids_to_tokens(self.gen_boi_token)
                    eoi = tokenizer.convert_ids_to_tokens(self.gen_eoi_token)
                    if (tokenizer.encode(boi, add_special_tokens=False) != [self.gen_boi_token]
                            or tokenizer.encode(eoi, add_special_tokens=False) != [self.gen_eoi_token]):
                        raise ValueError("BOI/EOI must encode as the configured single boundary IDs")
                    self._boundary_strings = (boi, eoi)
                boi, eoi = self._boundary_strings
                _text = f"{caption}{boi}{image.height} {image.width}{eoi}"
                # 哨兵只出现在 caption 那一段里，boi/eoi 的形状尾巴不含哨兵，整串一起
                # 过 _encode_structure_prompt 与分段处理等价。
                tokens = (
                    _encode_structure_prompt(_text, tokenizer)
                    if is_structure_caption
                    else encode_prompt(_text, tokenizer, None, "", "", split_quoted, quote_splitter)
                )
            elif keep_text:
                caption_ids = (
                    _encode_structure_prompt(caption, tokenizer)
                    if is_structure_caption
                    else encode_prompt(caption, tokenizer, None, "", "", split_quoted, quote_splitter)
                )
                tokens = torch.cat((caption_ids, torch.tensor(
                    [self.gen_boi_token, *shape_ids, self.gen_eoi_token], dtype=torch.int32,
                )))
            else:
                tokens = torch.tensor([self.gen_boi_token, *shape_ids, self.gen_eoi_token], dtype=torch.int32)
            n_shape = len(shape_ids)
            st = len(tokens) - n_shape - 2
            positions = torch.cat((
                build_text_token_type_ids(st + 1 + n_shape),
                build_text_token_type_ids(1, st + 2 + n_shape),
            ))
            input_ids = tokens[:-1].long()
            shape = vae_grid_thw[0].long()
            return dict(
                input_ids=input_ids, labels=torch.full_like(input_ids, -100),
                token_type_ids=positions[:-1].long(), loss_masks=torch.zeros_like(input_ids, dtype=torch.float32),
                seq_lens=torch.tensor([len(input_ids)]), attention_mask=torch.ones_like(input_ids,dtype=torch.bool),
                has_diff=True, source_name=item.get("source_name", ""), ds_idx=item.get("ds_idx", 0),
                vit_image_grid_thw=[shape], vae_pixel_values=[vae_pixel_values], vae_latent_shape=[shape],
                diff_pos_anchor=[torch.tensor(st + 1 + n_shape, dtype=torch.long)],
            )

        readable_text = f"<|user|><|gen_request|>{caption}<|assistant|><sop>{vae_latent_h} {vae_latent_w}<eop><|begin_of_image|><|vae|><|user|>"
        full_text = readable_text

        for vae_grid_thw_ in vae_grid_thw:
            num_vae_tokens = vae_grid_thw_.prod()
            full_text = full_text.replace("<|vae|>", "<|vae_placeholder|>" * num_vae_tokens, 1)
        full_text = full_text.replace("<|vae_placeholder|>", "<|vae|>", -1)

        # full_text / <|vae|> 占位那套逻辑完全不变，只有 text_render[*].text 那几段
        # 走结构专用的哨兵切分；普通 caption 仍按引号切（_DEFAULT_QUOTE_PAIRS）。
        tokens = (
            _encode_structure_prompt(full_text, tokenizer)
            if is_structure_caption
            else encode_prompt(full_text, tokenizer, None, "", "", split_quoted, quote_splitter)
        )

        position_ids = self.get_rope_index(
            input_ids=tokens.unsqueeze(0),
            image_grid_thw=vae_grid_thw,
            vae_grid_thw=vae_grid_thw,
        )

        loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)

        assert tokens.shape[0] == position_ids.shape[-1]

        # pre-shift (FFR convention)
        labels = tokens[1:].clone()
        shifted_loss_mask = loss_masks[1:]
        labels[shifted_loss_mask == 0] = -100
        loss_masks = shifted_loss_mask
        input_ids = tokens[:-1]
        position_ids = position_ids[..., :-1]

        # TODO(wwx): build block range, 用 tokenizer 里的特殊 token，其实应该传 transform，等 transform 的 token id 重构了删
        vae_token_id = tokenizer.encode("<|vae|>")[0]
        vae_mask = input_ids == vae_token_id
        block_range = self.mask_to_block_range(vae_mask)
        assert sum(vae_mask.to(torch.long)) == vae_grid_thw.prod(dim=-1).sum(), "vae token and grid mismatch"

        out_item = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "loss_masks": loss_masks,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "source_name": item.get("source_name", ""),
            "ds_idx": item.get("ds_idx", 0),
            "vit_image_grid_thw": torch.zeros((0, 3), dtype=torch.int32),
            "vit_pixel_values": torch.zeros((0, 3 * self.vit_patch_size**2), dtype=torch.float),
            "vae_pixel_values": [vae_pixel_values],
            "vae_grid_thw": vae_grid_thw,
            "block_range": block_range,
            "is_ref_vae": torch.tensor([False]),
            "text": readable_text,
        }
        return out_item

    def _process_edit(self, item: dict) -> dict:
        if self.template_mode == "pretrain_mode":
            from .wds_edit_transform import Glm4TransfusionEditTransform

            if not hasattr(self, "_edit_transform"):
                options = {f.name: getattr(self, f.name) for f in fields(Glm4TransfusionEditTransform)
                           if hasattr(self, f.name)}
                options.update(boi_token=self.gen_boi_token, eoi_token=self.gen_eoi_token,
                               image_token_id=self.gen_placeholder_token)
                self._edit_transform = Glm4TransfusionEditTransform(**options)
            row = dict(item)
            if "meta_info" not in row:
                meta = row.get("caption_info", {})
                row["meta_info"] = flatten_nested_dict(json.loads(meta) if isinstance(meta, str) else meta)
            return self._edit_transform(row)

        if self.tokenizer is None:
            raise ValueError("Glm4TransfusionImageTransform.tokenizer 未设置")

        if self.caption_pool:
            caption_keys, caption_probs = prepare_extra_texts_new(self.caption_pool)
            caption_info = json.loads(item["caption_info"])
            caption_info = flatten_nested_dict(caption_info)
            caption, caption_key_name = pick_text_with_key(caption_info, caption_keys, caption_probs)
        else:
            caption = ""
        if not caption:
            logger.warning_once("into legacy caption choice, please check your datasouce")
            raise ValueError(f"no caption key found in {item.get('__key__', '?')}")
        t2i_caption_list = caption_info[caption_key_name.replace("diff_caption", "image_captions", 1)]

        img_bytes_list = item["relation_info_binary"]
        images = [self._open_image(img_bytes) for img_bytes in img_bytes_list]
        images = [self._smart_resize(image, self.image_max_pixels) for image in images]
        vae_pixel_values = [normalize(to_tensor(image), 0.5, 0.5).unsqueeze(0) for image in images]
        vit_pixel_values = torch.cat(
            [
                rearrange(
                    vae_pixel_values_,
                    "b c (h ph) (w pw) -> (b h w) (c ph pw)",
                    ph=self.vit_patch_size,
                    pw=self.vit_patch_size,
                )
                for vae_pixel_values_ in vae_pixel_values[:-1]
            ],
            dim=0,
        )
        vae_grid_thw = torch.tensor(
            [[1, image.height // self.vae_latent_scale, image.width // self.vae_latent_scale] for image in images]
        )
        keep_text = not (self.text_dropout_rate > 0.0 and random.random() < self.text_dropout_rate)
        caption = caption if keep_text else ""

        readable_text = "<|user|><|gen_request|>"
        for i, vae_grid_thw_ in enumerate(vae_grid_thw[:-1, :], start=1):
            vae_latent_h = vae_grid_thw_[1]
            vae_latent_w = vae_grid_thw_[2]
            readable_text += (
                f"参考图{i}<sop>{vae_latent_h} {vae_latent_w}<eop><|begin_of_image|><|vae|><|image|><|end_of_image|>"
            )
        readable_text += caption
        readable_text += "<|assistant|>"
        if keep_text and random.random() < self.edit_with_t2i_caption_ratio:
            for i, t2i_caption in enumerate(t2i_caption_list[:-1], start=1):
                readable_text += f"参考图{i}: {t2i_caption}\n"
            readable_text += f"编辑后图像: {t2i_caption_list[-1]}"

        vae_latent_h = vae_grid_thw[-1, 1]
        vae_latent_w = vae_grid_thw[-1, 2]
        readable_text += f"<sop>{vae_latent_h} {vae_latent_w}<eop><|begin_of_image|><|vae|><|user|>"

        full_text = readable_text
        for vae_grid_thw_ in vae_grid_thw:
            num_vae_tokens = vae_grid_thw_.prod()
            full_text = full_text.replace("<|vae|>", "<|vae_placeholder|>" * num_vae_tokens, 1)
        full_text = full_text.replace("<|vae_placeholder|>", "<|vae|>", -1)

        for vae_grid_thw_ in vae_grid_thw[:-1, :]:
            num_vit_tokens = vae_grid_thw_.prod() // (self.vq_pre_pool_factor**2)
            full_text = full_text.replace("<|image|>", "<|image_placeholder|>" * num_vit_tokens, 1)
        full_text = full_text.replace("<|image_placeholder|>", "<|image|>")

        tokens = encode_prompt(full_text, self.tokenizer, None, "", "")

        position_ids = self.get_rope_index(
            input_ids=tokens.unsqueeze(0),
            image_grid_thw=vae_grid_thw[:-1, :],
            vae_grid_thw=vae_grid_thw,
        )

        loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)  # TODO(wwx): 这里应该 bool?

        assert tokens.shape[0] == position_ids.shape[-1]

        # pre-shift (FFR convention)
        labels = tokens[1:].clone()
        shifted_loss_mask = loss_masks[1:]
        labels[shifted_loss_mask == 0] = -100
        loss_masks = shifted_loss_mask
        input_ids = tokens[:-1]
        position_ids = position_ids[..., :-1]
        # TODO(wwx): build block range, 用 tokenizer 里的特殊 token，其实应该传 transform，等 transform 的 token id 重构了删
        vae_token_id = self.tokenizer.encode("<|vae|>")[0]
        vae_mask = input_ids == vae_token_id
        block_range_vae = self.mask_to_block_range(vae_mask)
        vit_token_id = self.tokenizer.encode("<|image|>")[0]
        vit_mask = input_ids == vit_token_id
        block_range_vit = self.mask_to_block_range(vit_mask)

        block_range = torch.cat([block_range_vae, block_range_vit], dim=0)
        block_range = block_range[block_range[:, 0].argsort()]
        assert sum(vae_mask.to(torch.long)) == vae_grid_thw.prod(dim=-1).sum(), "vae token and grid mismatch"
        assert sum(vit_mask.to(torch.long)) == vae_grid_thw[:-1, :].prod(dim=-1).sum() // (
            self.vq_pre_pool_factor**2
        ), "vit token and grid mismatch"

        out_item = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "loss_masks": loss_masks,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "source_name": item.get("source_name", ""),
            "ds_idx": item.get("ds_idx", 0),
            "vit_image_grid_thw": vae_grid_thw[:-1, ...],
            "vit_pixel_values": vit_pixel_values,
            "vae_pixel_values": vae_pixel_values,
            "vae_grid_thw": vae_grid_thw,
            "block_range": block_range,
            "is_ref_vae": torch.tensor([True] * (vae_grid_thw.shape[0] - 1) + [False]),
            "text": readable_text,
        }
        return out_item

    def _process_eval(self, item: dict) -> dict:
        if self.tokenizer is None:
            raise ValueError("Glm4TransfusionImageTransform.tokenizer 未设置")
        tokenizer = self.tokenizer
        vae_latent_scale = self.vae_latent_scale
        caption_key = self.caption_pool[0]["key"] if len(self.caption_pool) > 0 else "caption"

        caption = item[caption_key]
        is_calico_structure = self.eval_prompt_format == "calico_structure"
        if is_calico_structure:
            caption = _compact_structure_caption(caption, item.get("idx", item.get("__key__", "unknown")))
        elif _looks_like_structure_caption(caption):
            # 结构 caption 不带 eval_prompt_format 跑，整条 JSON 会当普通文本进 BPE：
            # 带着缩进空白和 \" 转义、text_render 的文字拿不到逐字符切分，和训练分布
            # 完全对不上 —— 出的图照样有，但评测结论是错的。宁可在这里停住。
            raise ValueError(
                f"Eval prompt (idx={item.get('idx')!r}, source={item.get('source_name', '')!r}) 是结构化 "
                "caption（JSON 带 objects/text_render），但该源没设 eval_prompt_format="
                "'calico_structure'，会按普通文本 BPE、和训练分布不一致。请在该源的 "
                "transform_kwargs 里加上 eval_prompt_format: calico_structure。"
            )
        ref_image_paths = item.get("ref_image_paths", [])

        if "test_h" in item and "test_w" in item:
            test_h = item["test_h"]
            test_w = item["test_w"]
        else:
            test_h = self.image_resize_max_length
            test_w = self.image_resize_max_length

        # image_list = [reference images (if any), generated-image placeholder].
        # Match training: reference images expose both <|vae|> and <|image|>;
        # the last/generated image exposes only <|vae|>.
        images = [self._open_image(ref_image_path) for ref_image_path in ref_image_paths]
        images.append(Image.new(mode="RGB", size=(test_w, test_h)))
        images = [self._smart_resize(image, self.image_max_pixels) for image in images]

        vae_pixel_values = [normalize(to_tensor(image), 0.5, 0.5).unsqueeze(0) for image in images]
        vit_pixel_values = (
            torch.cat(
                [
                    rearrange(
                        vae_pixel_values_,
                        "b c (h ph) (w pw) -> (b h w) (c ph pw)",
                        ph=self.vit_patch_size,
                        pw=self.vit_patch_size,
                    )
                    for vae_pixel_values_ in vae_pixel_values[:-1]
                ],
                dim=0,
            )
            if len(images) > 1
            else torch.zeros((0, 3 * self.vit_patch_size**2), dtype=torch.float)
        )  #  MARK: t2i 现在也要像训练一样创建空 tensor，因为有可能和 edit 混推

        vae_grid_thw = torch.tensor(
            [[1, image.height // vae_latent_scale, image.width // vae_latent_scale] for image in images]
        )
        image_grid_thw = vae_grid_thw[:-1, :]

        def build_input_ids_and_aux(prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
            readable_text = "<|user|><|gen_request|>"
            for i, vae_grid_thw_ in enumerate(image_grid_thw, start=1):
                vae_latent_h = vae_grid_thw_[1]
                vae_latent_w = vae_grid_thw_[2]
                readable_text += (
                    f"参考图{i}<sop>{vae_latent_h} {vae_latent_w}<eop>"
                    "<|begin_of_image|><|vae|><|image|><|end_of_image|>"
                )
            readable_text += prompt
            readable_text += "<|assistant|>"

            vae_latent_h = vae_grid_thw[-1, 1]
            vae_latent_w = vae_grid_thw[-1, 2]
            readable_text += f"<sop>{vae_latent_h} {vae_latent_w}<eop><|begin_of_image|><|vae|>"

            full_text = readable_text
            for vae_grid_thw_ in vae_grid_thw:
                num_vae_tokens = vae_grid_thw_.prod()
                full_text = full_text.replace("<|vae|>", "<|vae_placeholder|>" * num_vae_tokens, 1)
            full_text = full_text.replace("<|vae_placeholder|>", "<|vae|>", -1)

            for image_grid_thw_ in image_grid_thw:
                num_vit_tokens = image_grid_thw_.prod() // (self.vq_pre_pool_factor**2)
                full_text = full_text.replace("<|image|>", "<|image_placeholder|>" * num_vit_tokens, 1)
            full_text = full_text.replace("<|image_placeholder|>", "<|image|>")

            if is_calico_structure:
                try:
                    input_ids = _encode_structure_prompt(full_text, tokenizer)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSON escape in calico_structure eval prompt "
                        f"(idx={item.get('idx')!r}, source={item.get('source_name', '')!r}): {exc}"
                    ) from exc
            else:
                input_ids = encode_prompt(full_text, tokenizer, None, "", "")
            position_ids = self.get_rope_index(
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                vae_grid_thw=vae_grid_thw,
            )

            vae_token_id = tokenizer.encode("<|vae|>")[0]
            vit_token_id = tokenizer.encode("<|image|>")[0]
            vae_mask = input_ids == vae_token_id
            vit_mask = input_ids == vit_token_id
            block_range_vae = self.mask_to_block_range(vae_mask)
            block_range_vit = self.mask_to_block_range(vit_mask)
            if block_range_vit.numel() > 0:
                block_range = torch.cat([block_range_vae, block_range_vit], dim=0)
                block_range = block_range[block_range[:, 0].argsort()]
            else:
                block_range = block_range_vae

            assert sum(vae_mask.to(torch.long)) == vae_grid_thw.prod(dim=-1).sum(), "vae token and grid mismatch"
            assert sum(vit_mask.to(torch.long)) == image_grid_thw.prod(dim=-1).sum() // (self.vq_pre_pool_factor**2), (
                "vit token and grid mismatch"
            )
            assert input_ids.shape[0] == position_ids.shape[-1]
            return input_ids, position_ids, block_range, readable_text

        # 这里先按最经典的 dropout，只 dropout text 来，后面再看看有更特殊的需求怎么改（可能抽象不了函数）
        input_ids, position_ids, block_range, readable_text = build_input_ids_and_aux(caption)
        uncond_input_ids, uncond_position_ids, uncond_block_range, _ = build_input_ids_and_aux("")

        out_item = {
            "input_ids": input_ids,
            "uncond_input_ids": uncond_input_ids,
            "position_ids": position_ids,
            "uncond_position_ids": uncond_position_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "source_name": item.get("source_name", ""),
            "ds_idx": item.get("ds_idx", 0),
            "vit_image_grid_thw": image_grid_thw,
            "vit_pixel_values": vit_pixel_values,
            "vae_pixel_values": vae_pixel_values,
            "vae_grid_thw": vae_grid_thw,
            "block_range": block_range,
            "uncond_block_range": uncond_block_range,
            "is_ref_vae": torch.tensor([True] * len(ref_image_paths) + [False]),
            "idx": item["idx"],
            # 哨兵是实现细节，别让它进推理日志 / 出图的 EXIF prompt
            "prompt": readable_text.replace(_STRUCTURE_TEXT_START, "").replace(_STRUCTURE_TEXT_END, ""),
        }
        return out_item


# ══════════ 搬自 veomni/data/transforms/lyz_thinkgen_transform.py（整文件） ══════════

THINK_ID = 151350  # 旧 <think>  —— 新词表 decode='ऄ'（预期外观）

THINK_END_ID = 151351  # 旧 </think> —— 新词表 decode='अ'

# ── veomni/data/transforms/lyz_thinkgen_transform.py:79-98 ──
def _pick_text_with_key(r: dict, txt_keys: list, txt_probs) -> tuple[str, str]:
    """= 上游 `pick_text_with_key` + luyizhuo 0910 加固：字段存在但值为 ""/空白时
    视同缺失，从池中剔除并把概率重归一化摊给其余可用字段。

    为什么必须自带：上游版本只排除 None，artist 类源（上游打标偶发空串）会被选中
    空文本 → 下游按"无 caption"整样本误丢。实测这条加固救回的是真实可训样本。
    """
    txt_infos = [
        (i, key)
        for i, key in enumerate(txt_keys)
        if r.get(key, None) is not None and str(r.get(key)).strip() != ""
    ]
    if len(txt_infos) == 0:
        return "", ""
    txt_idx_, txt_keys_ = map(list, zip(*txt_infos))
    txt_probs_ = txt_probs[txt_idx_]
    txt_probs_ = txt_probs_ / txt_probs_.sum()
    txt_cumsum_ = np.cumsum(txt_probs_)
    idx = ((txt_cumsum_ - np.random.random()) < 0).sum()
    return r[txt_keys_[idx]], txt_keys_[idx]

# ── veomni/data/transforms/lyz_thinkgen_transform.py:101-116 ──
def _compute_resized_hw(width: int, height: int, max_pixels: int, scale_factor: int) -> tuple[int, int]:
    """纯函数复刻基类 `_smart_resize` 的尺寸计算（不触像素）。

    返回 (resized_h, resized_w)。与新仓 `Glm4ARImageTransform._smart_resize` 逐行同数学
    （该函数在新旧仓之间**逐字节相同**，已 diff 核对）：round 到 scale_factor 倍数、
    面积超 max_pixels 时按 sqrt 比例 floor 缩。ttg 实图路径与 to 无图路径共用，
    保证同一样本两模式的 H W 数字一致。
    """
    sf = scale_factor
    h_bar = max(sf, round(height / sf) * sf)
    w_bar = max(sf, round(width / sf) * sf)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / sf) * sf
        w_bar = math.floor(width / beta / sf) * sf
    return h_bar, w_bar

# ── veomni/data/transforms/lyz_thinkgen_transform.py:120-134 ──
def _chemfig_span(text: str):
    r"""从 caption 里取出 \chemfig{...} 全文（花括号配对扫描；体内有嵌套 {}）。"""
    i = text.find("\\chemfig{")
    if i < 0:
        return None
    depth = 0
    for k in range(i + len("\\chemfig"), len(text)):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i : k + 1]
    return None

_NAME_PREFIXES = (
    "A textbook-style structure diagram of ",
    "A chemical structure diagram of ",
    "A structural formula illustrating ",
    "A line-angle drawing of ",
    "The 2D skeletal formula of ",
    "Structure of the molecule ",
    "The structural formula of ",
    "Molecular structure of ",
    "Chemical structure of ",
    "Skeletal structure of ",
    "An image of ",
)

_NAME_SUFFIX = ", shown as a 2D structural formula"

_TRAIL_FORMULA = re.compile(r"\s*[(（][A-Za-z0-9+\-\[\]]+[)）]\s*$")

# ── veomni/data/transforms/lyz_thinkgen_transform.py:155-164 ──
def _name_payload(caption: str):
    """从 name 类 caption 里切出分子名载荷。定位不到返回 None（绝不瞎猜）。"""
    c = caption.strip()
    if c.endswith(_NAME_SUFFIX):
        return c[: -len(_NAME_SUFFIX)].strip() or None
    body = _TRAIL_FORMULA.sub("", c).strip()
    for pre in _NAME_PREFIXES:
        if body.startswith(pre):
            return body[len(pre) :].strip() or None
    return None

# ── veomni/data/transforms/lyz_thinkgen_transform.py:167-172 ──
def _smiles_payload(caption: str):
    """从 smiles 类 caption 里切出 SMILES 载荷（= 最后一个空白分隔 token）。"""
    toks = caption.strip().split()
    if len(toks) < 2:
        return None
    return toks[-1] or None

# ── veomni/data/transforms/lyz_thinkgen_transform.py:175-203 ──
def _quote_caption_payload(caption: str, cap_key: str, meta: dict) -> str:
    r"""把 caption 里的**载荷**用半角双引号包起来，使切分器判为 matched、从而逐字符编码。
    只包载荷、不包英文散文。载荷定位不到时原样返回；已带引号的不重复包装（幂等）。

    ★ 载荷本身可能含引号字符（IUPAC 撇位 2,2'- / 双撇位 N,N,N',N',N",N"-…）。
      若用载荷里已有的引号去包，切分器会在内部那个引号处提前闭合 → 只有前半被逐字符
      编码。解法：按偏好顺序挑一对**载荷里没出现过**的引号；都不可用则不包装。
    """
    if not caption:
        return caption
    k = (cap_key or "").lower()
    if "chemfig" in k:
        pay = _chemfig_span(caption)
    elif "smiles" in k:
        pay = _smiles_payload(caption)
    elif "name" in k:
        pay = _name_payload(caption)
    else:
        pay = None
    if not pay or pay not in caption:
        return caption
    for q1, q2 in (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’")):
        if q1 in pay or q2 in pay:
            continue
        wrapped = f"{q1}{pay}{q2}"
        if wrapped in caption:  # 幂等
            return caption
        return caption.replace(pay, wrapped, 1)
    return caption

# ── veomni/data/transforms/lyz_thinkgen_transform.py:206-660 ──
@dataclass
class Glm4TransfusionThinkGenTransform(Glm4TransfusionImageTransform):
    """三模式 transform。mode: "t2i" | "ttg" | "to"。其余字段继承基类语义。"""

    mode: str = "ttg"
    # 本类自建序列布局，不走基类的 template_mode 分支；写成 pretrain_mode 只为过基类
    # `__post_init__` 的取值校验（基类只认 sft_mode / pretrain_mode 两个值）。
    template_mode: str = "pretrain_mode"
    # 用户段分辨率控制 h:H w:W 的出现概率（拍板 §3-④：建议 0.9；∈{0,1} 时不消耗 RNG）
    resolution_prompt_prob: float = 0.9
    # ttg/to 的语言选择（缺该语言字段的样本自动回退另一侧；单语种配置不消耗 RNG）
    lang_pool: list = field(
        default_factory=lambda: [{"key": "en", "prob": 0.5}, {"key": "zh", "prob": 0.5}]
    )
    # t2i 请求模板池（默认=旧仓 120 条全量；[""] + resolution_prompt_prob=0 即基线回归模式）
    t2i_template_pool: list = field(default_factory=lambda: list(T2I_REQUEST_TEMPLATES))
    # 给 t2i 的 caption **载荷**加引号（化学对照实验用）。默认 False = 与既往 run 一致。
    caption_quote: bool = False
    # 引号抽取策略。"legacy" = 四个调用点全走旧叉乘正则；"cot_off" = **只关 CoT**，
    # raw prompt / final prompt / t2i caption 都照常抽取（用户 2026-10-14 拍板）。
    # 依据：实测 16 源 / 24,066 条 × 3 字段 —— CoT 三层全面最差（偏离 0.67% / 输入不平衡
    # 10.3% 且我们在其中 89% 仍强抽 / 抽到结构性散文 3.05%），而 raw 几乎干净。
    # 语义口径：**模型输出端 / 直接作画的那段**才抽（final、caption），
    #           **用户输入端 / 推理过程**不抽（raw、CoT）。
    quote_policy: str = "legacy"
    # 被抽取的那一段（final / caption）用哪套配对算法。"legacy"=旧叉乘正则；
    # "stack_tolerant"=★生产档（类型配对+栈匹配+Unicode 词内撇号保护+落单引号容错）。
    quote_impl: str = "legacy"
    # **只给 final 段单独指定算法**（留空=跟随 quote_impl）。为 calico_think_structured_*
    # 加的：它们的 final prompt 整条是 JSON，得用 "json_textrender"。
    quote_impl_final: str = ""
    # final prompt 引号不平衡时是否跳该样本（对照实验用，不建议开）。
    drop_if_final_quote_unbalanced: bool = False
    # 一条样本带**多套完整 CoT** 时的选择池。每项 = {"lang": "en"|"zh",
    # "fields": [prompt_key, thinking_key, final_key]}。选择语义：先按 lang_pool 的概率
    # 定语言，再在该语言的若干套里**等概率抽一整套**（不跨套混搭）；该语言全不完整时退到
    # 另一语言；都不行才 raise 跳样本。留空（默认）= 走 `_pick_lang_fields` 固定字段名路径。
    cot_pool: list = field(default_factory=list)
    # to 模式遇到 meta 缺 width/height 时：True（默认）= 丢样本；False = 退化路径
    # （截到 final 末 token、不出 <sop>H W<eop><boi> 段）。
    to_drop_if_no_dims: bool = True
    # ── 分辨率调度（旧仓里是改同事 `_sample_max_pixels`，这里改成子类 override）──
    #   "mix32k"   : 12.5% 限1024² / 12.5% [512²,1024²] / 25% [1024²,2048²] / 50% 限2048²
    #   "hi2k"     : 12.5% 限1024² / 29.1667% [1024²,2048²] / 58.3333% 限2048²（都 ≥1024²）
    #   "strict_2k": 固定 2048²（超 2k 缩到 2k、否则不变，不随机降）
    #   None       : 回落基类的 low_res_bias 逻辑（与同事口径一致）
    # ⚠ 面积上限只让 `_smart_resize` **缩**、不放大 ⇒ 源图本身 <1024² 的样本不会被
    #   hi2k 救起来，要配 `image_min_pixels_filter=1048576` 一起滤掉。
    res_sched: Optional[str] = None

    # ------------------------------------------------------------- helpers
    def _sample_max_pixels(self) -> int:
        """res_sched 优先于基类 low_res_bias；未设 res_sched 时逐字节回落基类实现。"""
        _P1K, _P512, _P2K = 1024 * 1024, 512 * 512, 2048 * 2048

        def _logu(lo: int, hi: int) -> int:
            return int(math.exp(random.uniform(math.log(lo), math.log(hi))))

        sched = self.res_sched or os.environ.get("VEOMNI_GEN_RES_SCHED")
        if sched == "mix32k":
            r = random.random()
            if r < 0.125:
                return _P1K  # 12.5% 限 1k
            elif r < 0.25:
                return _logu(_P512, _P1K)  # 12.5% [512²,1024²] log 均匀
            elif r < 0.50:
                return _logu(_P1K, _P2K)  # 25%  [1024²,2048²] log 均匀
            else:
                return _P2K  # 50% 限 2k
        if sched == "strict_2k":
            return _P2K
        if sched == "hi2k":
            # 三档，都 ≥1024²。推导：把 mix32k 里 <1024² 的那 12.5% 按原 25:50=1:2
            # 分给后两档 ⇒ 1/8 : 7/24 : 7/12。两个 env 旋钮（不设 = 推导值）：
            #   VEOMNI_GEN_RES_HI2K_P1K（默认 0.125）/ VEOMNI_GEN_RES_HI2K_P2K（默认 7/12）
            _p1k = float(os.environ.get("VEOMNI_GEN_RES_HI2K_P1K", 0.125))
            _p2k = float(os.environ.get("VEOMNI_GEN_RES_HI2K_P2K", 7.0 / 12.0))
            assert 0.0 <= _p1k and 0.0 <= _p2k and _p1k + _p2k <= 1.0 + 1e-9, (
                f"hi2k 的两个概率非法: P1K={_p1k} P2K={_p2k}（和必须 ≤ 1）"
            )
            _r = random.random()
            if _r < _p1k:
                return _P1K
            if _r < _p1k + _p2k:
                return _P2K
            return _logu(_P1K, _P2K)
        if sched is not None:
            raise ValueError(f"unknown res_sched {sched!r}; 可选 mix32k / hi2k / strict_2k")
        return super()._sample_max_pixels()

    # ── 按字段解析 quote_mode（见 quote_policy 字段注释）──
    def _qm_user(self) -> str:
        """user 段：t2i 是 caption、ttg/to 是 raw prompt —— **两者都要抽**。"""
        if self.quote_policy == "legacy":
            return "legacy"
        return self.quote_impl

    def _qm_think(self) -> str:
        """CoT 段：cot_off 下**不抽**。这是唯一关掉的位置。"""
        return "legacy" if self.quote_policy == "legacy" else "off"

    def _qm_final(self) -> str:
        """final 段：始终抽。quote_impl_final 非空时覆盖 quote_impl。"""
        if self.quote_policy == "legacy":
            return "legacy"
        if self.drop_if_final_quote_unbalanced and self.quote_impl != "legacy":
            return "stack_strict"
        return self.quote_impl_final or self.quote_impl

    def _enc(self, text: str, quote_mode: str) -> torch.Tensor:
        """新仓 `encode_prompt` 的官方扩展点：(split_quoted_text, quote_splitter)。"""
        split_quoted, splitter = resolve_splitter(quote_mode)
        return encode_prompt(text, self.tokenizer, None, "", "", split_quoted, splitter)

    def _pick_template(self) -> str:
        pool = self.t2i_template_pool or [""]
        if len(pool) == 1:
            return pool[0]  # 单模板不消耗 RNG（回归对齐前提）
        return random.choice(pool)

    def _draw_resolution_flag(self) -> bool:
        p = float(self.resolution_prompt_prob)
        if p <= 0.0:
            return False  # 不消耗 RNG
        if p >= 1.0:
            return True  # 不消耗 RNG
        return random.random() < p

    def _pick_cot_fields(self, meta: dict, key: str = "?") -> tuple[str, str, str]:
        """多套 CoT 的派发入口：配了 cot_pool 走新路径，否则原样走旧路径。

        `key` 只用于报错文案 —— loader 把 `key` 从 meta 里 pop 掉了，所以
        `meta.get('key')` 永远取不到；调用方传 `item['__key__']` 进来才能定位到样本。
        """
        if not self.cot_pool:
            return self._pick_lang_fields(meta, key)

        def take(entry):
            ks = entry["fields"] if isinstance(entry, dict) else entry[1]
            vals = []
            for k in ks:
                v = meta.get(k)
                v = str(v).strip() if v is not None else ""
                if not v:
                    return None
                vals.append(v)
            return tuple(vals)

        bylang: dict = {}
        for e in self.cot_pool:
            lg = e["lang"] if isinstance(e, dict) else e[0]
            bylang.setdefault(lg, []).append(e)

        langs = [d["key"] for d in self.lang_pool if d["key"] in bylang]
        if not langs:
            langs = list(bylang)
        if len(langs) == 1:
            order = [langs[0]]  # 单语种不消耗 RNG
        else:
            probs = {d["key"]: float(d["prob"]) for d in self.lang_pool}
            w = [probs.get(lg, 1.0) for lg in langs]
            r = random.random() * (sum(w) or 1.0)
            acc, first = 0.0, langs[-1]
            for lg, pb in zip(langs, w):
                acc += pb
                if r < acc:
                    first = lg
                    break
            order = [first] + [lg for lg in langs if lg != first]

        for lg in order:
            cands = list(bylang[lg])
            if len(cands) > 1:
                random.shuffle(cands)  # 同语言内等概率抽一套
            for e in cands:
                got = take(e)
                if got is not None:
                    return got
        raise ValueError(f"no complete cot set in cot_pool for langs={order} in {meta.get('key', key)!r}")

    def _pick_lang_fields(self, meta: dict, key: str = "?") -> tuple[str, str, str]:
        """按 lang_pool 选语言取 (prompt, thinking, final)，缺侧回退；两侧都不全 → raise 跳样本。"""

        def triple(lang: str):
            sfx = "" if lang == "en" else "_zh"
            p = meta.get("prompt" + sfx)
            t = meta.get("thinking_process" + sfx)
            f = meta.get("image_generation_prompt" + sfx)
            # 空白串视同缺失（否则「p=' '」会通过 if p 检查训出空 prompt）
            p, t, f = (str(x).strip() if x is not None else None for x in (p, t, f))
            if p and t and f:
                return p, t, f
            return None

        langs = [d["key"] for d in self.lang_pool]
        probs = [float(d["prob"]) for d in self.lang_pool]
        if len(langs) == 1:
            order = [langs[0]]  # 单语种不消耗 RNG
        else:
            s = sum(probs) or 1.0
            r = random.random() * s
            first = langs[-1]
            acc = 0.0
            for lg, pb in zip(langs, probs):
                acc += pb
                if r < acc:
                    first = lg
                    break
            order = [first] + [lg for lg in langs if lg != first]
        for lg in order:
            got = triple(lg)
            if got is not None:
                return got
        raise ValueError(f"no complete think triple for langs={langs} in {meta.get('key', key)!r}")

    def _user_prefix_tokens(self, body_text: str, res_h: Optional[int], res_w: Optional[int]):
        """[<|user|>, <|gen_image|>, enc(user_text)] + <|assistant|>；返回 (list[tensor], user_text)。

        拼接契约（训推一字不差）：user_text = body_text（模板已在 caller 拼好）；
        带分辨率时 += f" h:{H} w:{W}"（H/W = resize 后像素）。
        """
        user_text = body_text
        if res_h is not None:
            user_text = f"{user_text} h:{res_h} w:{res_w}"
        toks = self._enc(user_text, self._qm_user())
        user_tok = torch.tensor([self.user_token_id], dtype=torch.int32)
        gen_tok = torch.tensor([self.gen_image_request_token_id], dtype=torch.int32)
        asst_tok = torch.tensor([self.assistant_token_id], dtype=torch.int32)
        return [user_tok, gen_tok, toks, asst_tok], user_text

    def _assistant_think_tokens(
        self, thinking: str, final: str, grid_h: Optional[int], grid_w: Optional[int]
    ) -> torch.Tensor:
        """loss=1 的 assistant 目标段（不含 <boi>）:
        [151350] + enc(思考链) + [151351] + enc(f"{final} ") + enc(f"<sop>{Hg} {Wg}<eop>")
        —— final 与 <sop> 之间恰好 1 个空格（拍板 §4 版式）。
        grid 为 None（退化：无宽高）时省略 shape 段与尾空格。
        """
        think_open = torch.tensor([THINK_ID], dtype=torch.int32)
        think_close = torch.tensor([THINK_END_ID], dtype=torch.int32)
        body = self._enc(thinking, self._qm_think())
        if grid_h is None:
            final_toks = self._enc(final, self._qm_final())
            return torch.cat(
                [think_open, body.to(torch.int32), think_close, final_toks.to(torch.int32)], dim=0
            )
        final_toks = self._enc(f"{final} ", self._qm_final())
        # shape 段永远不做引号切分（纯数字 + 两个控制 token）
        shape_toks = self._enc(f"<sop>{grid_h} {grid_w}<eop>", "legacy")
        return torch.cat(
            [
                think_open,
                body.to(torch.int32),
                think_close,
                final_toks.to(torch.int32),
                shape_toks.to(torch.int32),
            ],
            dim=0,
        )

    @staticmethod
    def _preshift_pack(
        tokens: torch.Tensor, loss_masks: torch.Tensor, token_type_ids: torch.Tensor
    ) -> dict:
        """FFR 式 pre-shift（与基类逐行同构）：input_ids=tokens[:-1]，labels=tokens[1:]，
        mask 标注在未 shift 的「目标 token」上 → shift 后自然得到正确的 -100 分布。"""
        assert tokens.shape[0] == token_type_ids.shape[0] == loss_masks.shape[0]
        labels = tokens[1:].clone().to(torch.long)
        shifted = loss_masks[1:]
        labels[shifted == 0] = -100
        input_ids = tokens[:-1].to(torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids[:-1].to(torch.long),
            "loss_masks": shifted,
            "seq_lens": torch.tensor([input_ids.shape[0]]),
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }

    # ------------------------------------------------------------- __call__
    def __call__(self, item: dict) -> dict | None:
        if self.tokenizer is None:
            raise ValueError("Glm4TransfusionThinkGenTransform.tokenizer 未设置")
        if self.mode not in ("t2i", "ttg", "to"):
            raise ValueError(f"unknown mode {self.mode!r}")

        meta_info = item.get("meta_info")
        meta = json.loads(meta_info) if isinstance(meta_info, str) else (meta_info or {})

        if self.mode == "to":
            return self._call_to(item, meta)
        return self._call_gen(item, meta)  # t2i / ttg（都带扩散分支）

    # ---- t2i / ttg：读图 + 扩散字段（结构与基类 _process_t2i(pretrain_mode) 对齐）----
    def _call_gen(self, item: dict, meta: dict) -> dict:
        # 1) 文本素材
        if self.mode == "t2i":
            _cap_key = ""
            if self.caption_pool:
                txt_keys, txt_probs = prepare_extra_texts_new(self.caption_pool)
                caption, _cap_key = _pick_text_with_key(meta, txt_keys, txt_probs)
            else:
                caption = ""
            if not caption:
                fb = [
                    k
                    for k in (
                        "long_caption_zh",
                        "long_caption_en",
                        "mid_caption_zh",
                        "mid_caption_en",
                        "short_caption_zh",
                        "short_caption_en",
                    )
                    if meta.get(k) is not None
                ]
                if not fb:
                    raise ValueError(f"no caption key found in {item.get('__key__', '?')}")
                caption = meta[random.choice(fb)]
            caption = (caption if isinstance(caption, str) else str(caption)).strip()
            if not caption:
                raise ValueError(f"no caption key found in {item.get('__key__', '?')} (empty value)")
            if self.caption_quote:
                caption = _quote_caption_payload(caption, _cap_key, meta)
            template = self._pick_template()
            body_text = f"{template} {caption}" if template else caption
            thinking = final = None
        else:  # ttg
            prompt, thinking, final = self._pick_cot_fields(meta, item.get("__key__", "?"))
            body_text = f"{THINKING_TEMPLATE} {prompt}"

        # 2) 图像（与基类同管线：open→filter→smart_resize→[-1,1]）
        img_bytes = None
        for image_key in ["file_path_binary", *self.image_keys]:
            if item.get(image_key) is not None:
                img_bytes = item[image_key]
                break
        if img_bytes is None:
            raise ValueError(f"no image bytes found in {item.get('__key__', '?')}")
        image = self._open_image(img_bytes)
        if self.image_min_pixels_filter is not None:
            w, h = image.size
            if w * h < self.image_min_pixels_filter:
                raise ValueError(
                    f"image area {w * h} < image_min_pixels_filter {self.image_min_pixels_filter}"
                )
        if self.max_ratio is not None:
            w, h = image.size
            if max(w, h) / max(min(w, h), 1) > self.max_ratio:
                raise ValueError(f"aspect ratio > max_ratio {self.max_ratio}")
        image = self._smart_resize(image, max_pixels=self._sample_max_pixels())
        image_tensor = normalize(to_tensor(image), 0.5, 0.5)
        vae_pixel_values = image_tensor.unsqueeze(0)
        res_h, res_w = image.height, image.width  # 用户段 h:w = resize 后像素
        grid_h = res_h // self.vae_latent_scale  # <sop> 网格 = 像素//16，同源派生
        grid_w = res_w // self.vae_latent_scale
        vae_latent_shape = torch.tensor([1, grid_h, grid_w])

        boi_tok = torch.tensor([self.gen_boi_token], dtype=torch.int32)
        user_tok = torch.tensor([self.user_token_id], dtype=torch.int32)

        # 3) CFG condition dropout（与基类同语义：最小 <|user|> 对，loss 全 0，扩散保留）
        keep_text = not (self.text_dropout_rate > 0.0 and random.random() < self.text_dropout_rate)
        if not keep_text:
            tokens = torch.cat([user_tok, user_tok.clone()], dim=0)
            token_type_ids = torch.cat(
                [build_text_token_type_ids(1, st_idx=0), build_text_token_type_ids(1, st_idx=1)], dim=0
            )
            diff_pos_anchor = 2
            loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)
        else:
            with_res = self._draw_resolution_flag()
            prefix_parts, _ = self._user_prefix_tokens(
                body_text, res_h if with_res else None, res_w if with_res else None
            )
            if self.mode == "t2i":
                shape_toks = self._enc(f"<sop>{grid_h} {grid_w}<eop>", "legacy")
                target_parts = [shape_toks.to(torch.int32), boi_tok]
                target_mask_val = 0.0  # 纯 T2I：整条 AR 是扩散条件（labels 全 -100）
            else:
                asst_target = self._assistant_think_tokens(thinking, final, grid_h, grid_w)
                target_parts = [asst_target, boi_tok]
                target_mask_val = 1.0  # TTG：think+final+shape+<boi> 全部是预测目标
            tail_user = user_tok.clone()
            parts = prefix_parts + target_parts + [tail_user]
            tokens = torch.cat(parts, dim=0)
            n_prefix = sum(p.shape[0] for p in prefix_parts)
            n_target = sum(p.shape[0] for p in target_parts)
            loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)
            loss_masks[n_prefix : n_prefix + n_target] = target_mask_val
            # 尾 <|user|> 恒 0 → pre-shift 后 <boi> 位置 label=-100（boi 后物理接扩散块）
            token_type_ids = build_text_token_type_ids(tokens.shape[0], st_idx=0)
            diff_pos_anchor = int(tokens.shape[0] - 1)  # <boi>（倒数第2）+1 = 尾user位 = len-1

        out = self._preshift_pack(tokens, loss_masks, token_type_ids)
        out.update(
            {
                "has_diff": True,
                "source_name": item.get("source_name", ""),
                "ds_idx": item.get("ds_idx", 0),
                "vit_image_grid_thw": [vae_latent_shape.to(torch.long)],
                "vae_pixel_values": [vae_pixel_values],
                "vae_latent_shape": [vae_latent_shape],
                "diff_pos_anchor": [torch.tensor(diff_pos_anchor, dtype=torch.long)],
            }
        )
        return out

    # ---- to：纯文本，不读图 ----
    def _call_to(self, item: dict, meta: dict) -> dict:
        prompt, thinking, final = self._pick_cot_fields(meta, item.get("__key__", "?"))
        body_text = f"{THINKING_TEMPLATE} {prompt}"

        mw, mh = meta.get("width"), meta.get("height")
        # 宽高偶发脏值（"600.5"/负数/非数字）：解析失败或非正 → 走无宽高路径
        try:
            mw, mh = (int(float(mw)), int(float(mh))) if (mw and mh) else (None, None)
            if mw is not None and (mw <= 0 or mh <= 0):
                mw = mh = None
        except (TypeError, ValueError):
            mw = mh = None
        if mw and mh:
            # 与 ttg 实图路径同一过滤与 resize 数学（保证 H W 数字两模式一致）
            if self.image_min_pixels_filter is not None and mw * mh < self.image_min_pixels_filter:
                raise ValueError(f"meta area {mw * mh} < image_min_pixels_filter")
            if self.max_ratio is not None and max(mw, mh) / max(min(mw, mh), 1) > self.max_ratio:
                raise ValueError("aspect ratio > max_ratio (meta)")
            res_h, res_w = _compute_resized_hw(mw, mh, self._sample_max_pixels(), self.scale_factor)
            grid_h, grid_w = res_h // self.vae_latent_scale, res_w // self.vae_latent_scale
        elif self.to_drop_if_no_dims:
            raise ValueError(
                f"no width/height in meta for to-mode {meta.get('key', item.get('__key__', '?'))!r}"
            )
        else:
            res_h = res_w = grid_h = grid_w = None

        with_res = self._draw_resolution_flag() if res_h is not None else False
        prefix_parts, _ = self._user_prefix_tokens(
            body_text, res_h if with_res else None, res_w if with_res else None
        )
        asst_target = self._assistant_think_tokens(thinking, final, grid_h, grid_w)
        parts = list(prefix_parts) + [asst_target]
        if grid_h is not None:
            # 正常路径：tokens 止于 <boi>（无尾 user、无 EOS）→ pre-shift 后末目标=预测 <boi>
            parts.append(torch.tensor([self.gen_boi_token], dtype=torch.int32))
        tokens = torch.cat(parts, dim=0)
        n_prefix = sum(p.shape[0] for p in prefix_parts)
        loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)
        loss_masks[n_prefix:] = 1.0  # think..final..[shape..<boi>] 全为目标
        token_type_ids = build_text_token_type_ids(tokens.shape[0], st_idx=0)

        out = self._preshift_pack(tokens, loss_masks, token_type_ids)
        out.update({"source_name": item.get("source_name", ""), "ds_idx": item.get("ds_idx", 0)})
        return out  # 纯文本最小集：无 has_diff / vae / vit 字段（collator 走 packed 分支的 else）

__all__ = [
    "Glm4TransfusionThinkGenTransform",
    "THINK_ID",
    "THINK_END_ID",
    "_compute_resized_hw",
]


# ══════════ 搬自 veomni/data/transforms/lyz_thinkgen_infer.py（整文件） ══════════

SOP_ID = 154824  # <sop>

EOP_ID = 154825  # <eop>

BOI_ID = 154830  # <boi> = <|begin_of_image|> = image_start_token_id

# ── veomni/data/transforms/lyz_thinkgen_infer.py:63-203 ──
class ThinkGenInferenceTransform(Glm4TransfusionThinkGenTransform):
    """训练 transform 的推理孪生体。构造参数与训练完全一样（见模块 docstring）。"""

    # ---------------------------------------------------------------- 几何
    def check_resolution(self, height: int, width: int) -> None:
        """H/W 必须同时满足：`scale_factor`（resize 对齐，训练是 32）与
        `vae_latent_scale`（patchified VAE 步长 16）的整数倍。

        不满足就硬错 —— 训练侧会被 `_smart_resize` 悄悄改成对齐值，推理侧若放过去，
        出来的图尺寸与请求不符而且不报错。
        """
        for name, stride in (("scale_factor", int(self.scale_factor)),
                             ("vae_latent_scale", int(self.vae_latent_scale))):
            if height % stride or width % stride:
                raise ValueError(
                    f"目标分辨率 {height}x{width} 不是 {name}={stride} 的整数倍 —— "
                    "训练侧的 _smart_resize 会改掉它，推理侧必须由调用方给对齐值"
                )

    def grid_of(self, height: int, width: int) -> tuple[int, int]:
        return height // int(self.vae_latent_scale), width // int(self.vae_latent_scale)

    # ------------------------------------------------------------ 文本素材
    def user_body_text(self, prompt: str, mode: str, idx: int = 0) -> tuple[str, str]:
        """返回 (body_text, template)。模板选择对同一个 prompt 稳定（只由 idx 决定），
        这样同一条 prompt 的不同分辨率档位共享同一个模板，横比才干净。"""
        if mode == "ttg":
            return f"{THINKING_TEMPLATE} {prompt}", THINKING_TEMPLATE
        pool = self.t2i_template_pool or [""]
        if len(pool) == 1:
            template = pool[0]
        else:
            # 固定 salt + idx：与训练的 random.choice 同分布，但可复现、与分辨率无关。
            template = random.Random(0x7117 + int(idx)).choice(pool)
        return (f"{template} {prompt}" if template else prompt), template

    # ------------------------------------------------------- Stage A 前缀
    def ar_prefix_ids(self, prompt: str, height: int, width: int, *, mode: str = "ttg",
                      idx: int = 0, with_res: bool = True) -> tuple[torch.Tensor, str]:
        """`<|user|> <|gen_image|> enc(user_text) <|assistant|>` 的 token 序列 [L]。

        用的是训练那同一个 `_user_prefix_tokens`（含 `_qm_user()` 的引号档位），
        所以 Stage A 的前缀与训练时模型见过的前缀**逐 token 相同**。
        """
        self.check_resolution(height, width)
        body_text, _ = self.user_body_text(prompt, mode, idx)
        parts, user_text = self._user_prefix_tokens(
            body_text, height if with_res else None, width if with_res else None
        )
        return torch.cat(parts, dim=0).to(torch.long), user_text

    def shape_tail_ids(self, height: int, width: int) -> torch.Tensor:
        """`enc("<sop>Hg Wg<eop>")` + `<boi>` —— 与训练的 shape 段逐 token 相同。"""
        grid_h, grid_w = self.grid_of(height, width)
        shape = self._enc(f"<sop>{grid_h} {grid_w}<eop>", "legacy").to(torch.int32)
        boi = torch.tensor([self.gen_boi_token], dtype=torch.int32)
        return torch.cat([shape, boi], dim=0).to(torch.long)

    # -------------------------------------------------------- Stage B 特征
    def _blank_target(self, height: int, width: int) -> torch.Tensor:
        """H×W 的灰底图，按训练同一条像素管线归一化到 [-1, 1]。

        它只是 `vae_latent_shape` / `vit_image_grid_thw` 的形状载体：
        `prepare_inference_batch` 只把 `is_ref_vae=True` 的行送进 VAE，目标行是 zeros，
        真正的起点由 `_denoise_and_decode` 里的 `torch.randn` 给。
        """
        blank = Image.new("RGB", (width, height), (127, 127, 127))
        return normalize(to_tensor(blank), 0.5, 0.5).unsqueeze(0)

    def _pack(self, tokens: torch.Tensor, anchor: int, vae_latent_shape: torch.Tensor,
              vae_pixel_values: torch.Tensor) -> dict:
        loss_masks = torch.zeros(tokens.shape[0], dtype=torch.float32)  # 推理不算 AR loss
        token_type_ids = build_text_token_type_ids(tokens.shape[0], st_idx=0)
        out = self._preshift_pack(tokens, loss_masks, token_type_ids)
        out.update(
            {
                "has_diff": True,
                "source_name": "",
                "ds_idx": 0,
                "vit_image_grid_thw": [vae_latent_shape.to(torch.long)],
                "vae_pixel_values": [vae_pixel_values],
                "vae_latent_shape": [vae_latent_shape],
                "diff_pos_anchor": [torch.tensor(anchor, dtype=torch.long)],
            }
        )
        return out

    def _uncond_feature(self, vae_latent_shape: torch.Tensor,
                        vae_pixel_values: torch.Tensor) -> dict:
        """逐行复刻训练侧 `text_dropout_rate` 命中时那条分支
        （`lyz_thinkgen_transform.py:567-574`）：tokens=[user,user] / anchor=2 / loss 全 0。

        ⚠ 不要换成同事的 `make_unconditional()`（保留 `<boi>` 之后的部分）：
        那是他 t2i 布局下的做法，而**我们这个 ckpt 的模型在训练里见过的"无条件"是这个**。
        """
        user_tok = torch.tensor([self.user_token_id], dtype=torch.int32)
        tokens = torch.cat([user_tok, user_tok.clone()], dim=0)
        return self._pack(tokens, 2, vae_latent_shape, vae_pixel_values)

    def build_feature(self, *, prompt: str, height: int, width: int, mode: str,
                      gen_ids: list[int] | None = None, idx: int = 0,
                      with_res: bool = True) -> dict:
        """造出送进 `MyCollator` 的那个 item。

        mode="t2i"  : gen_ids 必须为 None（不思考、AR 段只有 shape+boi）
        mode="ttg"  : gen_ids = Stage A 生成的 token（**已截到 `<sop>` 之前**，不含 boi）
        """
        if mode not in ("t2i", "ttg"):
            raise ValueError(f"推理只支持 t2i / ttg，得到 {mode!r}")
        if mode == "t2i" and gen_ids:
            raise ValueError("t2i(direct) 模式不接受生成 token")
        self.check_resolution(height, width)
        grid_h, grid_w = self.grid_of(height, width)
        vae_latent_shape = torch.tensor([1, grid_h, grid_w])
        vae_pixel_values = self._blank_target(height, width)

        body_text, _template = self.user_body_text(prompt, mode, idx)
        prefix_parts, user_text = self._user_prefix_tokens(
            body_text,
            height if with_res else None,
            width if with_res else None,
        )
        target_parts: list[torch.Tensor] = []
        if mode == "ttg":
            if gen_ids:
                target_parts.append(torch.tensor(list(gen_ids), dtype=torch.int32))
        shape_toks = self._enc(f"<sop>{grid_h} {grid_w}<eop>", "legacy").to(torch.int32)
        target_parts.append(shape_toks)
        target_parts.append(torch.tensor([self.gen_boi_token], dtype=torch.int32))
        tail_user = torch.tensor([self.user_token_id], dtype=torch.int32)

        tokens = torch.cat([*prefix_parts, *target_parts, tail_user], dim=0)
        anchor = int(tokens.shape[0] - 1)  # = len(input_ids)（<boi> 之后的扩散块锚点）
        feature = self._pack(tokens, anchor, vae_latent_shape, vae_pixel_values)
        feature["_uncond_feature"] = self._uncond_feature(vae_latent_shape, vae_pixel_values)
        feature["idx"] = int(idx)
        feature["prompt"] = prompt
        feature["_user_text"] = user_text
        feature["_n_prefix"] = int(sum(p.shape[0] for p in prefix_parts))
        feature["_n_shape"] = int(shape_toks.shape[0])
        return feature

# ── veomni/data/transforms/lyz_thinkgen_infer.py:206-217 ──
def infer_base_transform_kwargs(tokenizer, model_config, seed: int = 0) -> dict:
    """= 训练 `build_interleave_dataset_from_configs(base_transform_kwargs=...)` 的 HY 分支
    （`tasks/transfusion/train.py:2223-2229`）。少了 `tokenizer` 这条，factory 造出来的
    transform 的 `tokenizer` 就是 None，第一次 `_enc` 才报 `'NoneType' is not callable`。
    """
    vision = model_config.vq_vision_config
    return {
        "tokenizer": tokenizer,
        "seed": seed,
        "vit_patch_size": vision.patch_size,
        "vq_pre_pool_factor": vision.vq_pre_pool_factor,
    }

# ── veomni/data/transforms/lyz_thinkgen_infer.py:220-254 ──
def build_inference_transform(tokenizer, model_config, transform_kwargs: dict, mode: str = "ttg",
                              seed: int = 0):
    """用**训练那条 factory** 把 options 配好，再把字段搬到推理子类上。

    与 `tasks/transfusion/train_sft_lyz.py::_build_thinkgen` 同一套做法（同一个
    `make_hy_transform_factory` → `wds_image_with_meta_json` 分支 → 字段搬运），
    所以 geometry（`vae_latent_scale` / `latent_scale` / `cond_merge_size`）、三个
    角色 token id 的查表与 fallback、按 dataclass 字段过滤，全部复用同事的实现。
    """
    from dataclasses import fields

    from vllm_omni.diffusion.models.glm_image_moe_transfusion.vendor.model_impl.pipeline import make_hy_transform_factory

    factory = make_hy_transform_factory(
        tokenizer=tokenizer,
        ffr_image_config=None,
        image_token_id=model_config.image_token_id,
        image_start_token_id=model_config.image_start_token_id,
        image_end_token_id=model_config.image_end_token_id,
        lang_vocab_size=model_config.text_config.vocab_size,
        vq_compression_ratio=model_config.vq_compression_ratio,
        vae_compression_ratio=model_config.vae_compression_ratio,
        vae_token_id=getattr(model_config, "vae_token_id", None),
        native_vision_config=(model_config if getattr(model_config, "hy_native_vision", False) else None),
        vae_token_stride=model_config.vae_compression_ratio,
        cond_merge_size=model_config.cond_merge_size,
    )
    options = {**infer_base_transform_kwargs(tokenizer, model_config, seed), **dict(transform_kwargs)}
    wrapped = factory(dict(options), "wds_image_with_meta_json")
    base = getattr(wrapped, "transform", wrapped)
    values = {f.name: getattr(base, f.name) for f in fields(type(base))}
    own = {f.name for f in fields(Glm4TransfusionThinkGenTransform)} - set(values)
    values.update({k: v for k, v in options.items() if k in own})
    values["mode"] = mode
    return ThinkGenInferenceTransform(**values)

__all__ = ["ThinkGenInferenceTransform", "build_inference_transform",
           "infer_base_transform_kwargs", "SOP_ID", "EOP_ID", "BOI_ID"]


# ══════════ 搬自 veomni/data/data_collator.py（闭包 1/10 个定义） ══════════

default_collate_fn_map[type(None)] = default_collate_fn_map[str]

# ── veomni/data/data_collator.py:65-76 ──
@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...



# ════════════════════════════════════════════════════════════════
#  cu_seqlens / position_ids 的互转 —— 逐字抄自 `veomni/utils/seqlen_pos_transform_utils.py`
#  （2026-11-17 从单独的 processing_hy_seqlen.py 并进来）
# ════════════════════════════════════════════════════════════════

def len2culen(seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the sequence lengths to cumulative sequence lengths.

    NOTE: flash attention only accepts int32 cu_seqlens.
    """
    return F.pad(torch.cumsum(seqlens, dim=0), (1, 0)).type(torch.int32)


def culen2len(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the cumulative sequence lengths to sequence lengths.
    """
    return cu_seqlens.diff()


def pos2culen(position_ids: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the position ids to cumulative sequence lengths.
    """
    if position_ids.dim() == 3:  # (batch_size, dim, seq_length):
        position_ids = position_ids[:, 0, :]

    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), dtype=torch.int32, device=position_ids.device)
    return F.pad(indices_q[position_ids == 0], (0, 1), "constant", position_ids.size(0))


def culen2pos(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the cumulative sequence lengths to position ids.
    """
    seqlens = culen2len(cu_seqlens).cpu()
    position_ids = torch.cat([torch.arange(length, dtype=torch.long, device=cu_seqlens.device) for length in seqlens])
    return position_ids.unsqueeze(0)


def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    Copy from transformers/modeling_flash_attention_utils.py 354567d955fbc5fbd70fc841b7a7bcc654bea3f1
    This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids.

    Arguments:
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length_q = cu_seq_lens_q.diff().max()
    # NOTE: With torch compile, this will cause a graph break if you don't set
    # `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` in the environment or call
    # `torch._dynamo.config.capture_scalar_outputs = True` before doing the forward pass.
    # This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
    # requires `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`.
    max_length_q = max_length_q.item()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)


