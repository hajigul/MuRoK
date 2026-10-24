"""Frozen multimodal generator conditioned on retrieved triplets.

Default: Qwen/Qwen2.5-VL-7B-Instruct — open weights from HuggingFace,
NO API key needed, fits on a single 24 GB GPU in bfloat16 (~16-17 GB).
If you need more headroom (long prompts + many query images), pass
--load-4bit to quantize with bitsandbytes (~7 GB).

Text-only fallback: Qwen/Qwen2.5-7B-Instruct (use --text-only if your
queries carry no images or you want a lighter run).
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .data import MKG, Query

VLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a question answering assistant. Use the retrieved knowledge "
    "graph triplets (and images, if provided) to answer the question. "
    "Answer with the short answer only — no explanation."
)


def format_evidence(mkg: MKG, triplet_indices: List[int]) -> str:
    lines = []
    for rank, i in enumerate(triplet_indices, 1):
        c = mkg.triplets[i]
        lines.append(f"{rank}. ({c.h_name}, {c.r_name}, {c.t_name})")
    return "Retrieved knowledge:\n" + "\n".join(lines)


class QwenVLGenerator:
    def __init__(self, load_4bit: bool = False, model_name: str = VLM_NAME):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            kwargs.pop("torch_dtype")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def answer(self, query: Query, evidence: str, evidence_images: Optional[List[str]] = None) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        if query.image is not None:
            content.append({"type": "image", "image": query.image})
        for p in evidence_images or []:
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": f"{evidence}\n\nQuestion: {query.question}\nShort answer:"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        out = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


class QwenTextGenerator:
    def __init__(self, load_4bit: bool = False, model_name: str = LLM_NAME):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            kwargs.pop("torch_dtype")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def answer(self, query: Query, evidence: str, evidence_images=None) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{evidence}\n\nQuestion: {query.question}\nShort answer:"},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        out = out[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


# --------------------------------------------------------------- gen metrics
def _norm(s: str) -> str:
    import re, string

    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> float:
    p = _norm(pred)
    return float(any(p == _norm(g) for g in golds))


def token_f1(pred: str, golds: List[str]) -> float:
    from collections import Counter

    p_toks = _norm(pred).split()
    best = 0.0
    for g in golds:
        g_toks = _norm(g).split()
        common = Counter(p_toks) & Counter(g_toks)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p_toks), n / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def contains_at_1(pred: str, golds: List[str]) -> float:
    p = _norm(pred)
    return float(any(_norm(g) in p for g in golds if _norm(g)))


def bleu1(pred: str, golds: List[str]) -> float:
    """Unigram BLEU with brevity penalty against the closest reference."""
    import math
    from collections import Counter

    p_toks = _norm(pred).split()
    if not p_toks:
        return 0.0
    best = 0.0
    for g in golds:
        g_toks = _norm(g).split()
        if not g_toks:
            continue
        overlap = sum((Counter(p_toks) & Counter(g_toks)).values())
        prec = overlap / len(p_toks)
        bp = 1.0 if len(p_toks) >= len(g_toks) else math.exp(1 - len(g_toks) / len(p_toks))
        best = max(best, bp * prec)
    return best


def evaluate_generation(preds: List[str], gold_lists: List[List[str]]):
    n = len(preds)
    return {
        "EM": 100 * sum(exact_match(p, g) for p, g in zip(preds, gold_lists)) / n,
        "F1": 100 * sum(token_f1(p, g) for p, g in zip(preds, gold_lists)) / n,
        "Contains@1": 100 * sum(contains_at_1(p, g) for p, g in zip(preds, gold_lists)) / n,
        "BLEU-1": 100 * sum(bleu1(p, g) for p, g in zip(preds, gold_lists)) / n,
    }
