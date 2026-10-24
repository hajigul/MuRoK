"""Retrieval metrics: NDCG@K, Precision@K, Recall@K (binary relevance)."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def ndcg_at_k(ranked: Sequence[int], gold: set, k: int) -> float:
    dcg = 0.0
    for i, idx in enumerate(ranked[:k]):
        if idx in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def precision_at_k(ranked: Sequence[int], gold: set, k: int) -> float:
    hits = sum(1 for idx in ranked[:k] if idx in gold)
    return hits / k


def recall_at_k(ranked: Sequence[int], gold: set, k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for idx in ranked[:k] if idx in gold)
    return hits / len(gold)


def evaluate_retrieval(
    rankings: List[List[int]],
    gold_sets: List[set],
    ks: Sequence[int] = (5, 10, 20, 50, 100),
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(rankings)
    for k in ks:
        out[f"NDCG@{k}"] = 100 * sum(ndcg_at_k(r, g, k) for r, g in zip(rankings, gold_sets)) / n
        out[f"P@{k}"] = 100 * sum(precision_at_k(r, g, k) for r, g in zip(rankings, gold_sets)) / n
        out[f"R@{k}"] = 100 * sum(recall_at_k(r, g, k) for r, g in zip(rankings, gold_sets)) / n
    return out
