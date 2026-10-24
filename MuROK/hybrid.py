"""HybridMorseRetriever: dispatches each query to whichever MORSE
architecture wins for its modality, rather than forcing one model to
serve both regimes.

  - query has NO resolvable image  -> legacy 3-expert MORSE (morse_legacy.py),
    which empirically wins the text-only settings (S2/S3) on FB15k-237-IMG.
  - query HAS a resolvable image    -> current 5-expert MORSE (morse.py),
    which empirically wins the image-rich settings (S1/S4/S5).

The dispatch rule is `Query.has_visual_evidence` (i.e. delta_q), the same
ground-truth signal already used everywhere else in this codebase for
"does this query actually carry an image" -- not the benchmark's
`is_multimodal` pool-membership flag, which can be True even when the
image file is missing.

Trade-off, stated plainly: this is two trained models plus an un-learned
switch, not one unified router. It should out-perform either single
checkpoint on the full query mix, but the routing decision here is a fixed
rule, not something learned end-to-end the way expert weighting is within
each of the two models. Document this explicitly in the paper if used for
headline numbers -- see the printed rationale in build_hybrid().
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .data import MKG, Query
from .index import TripletIndex
from .morse import MorseRetriever, MorseScorer
from .morse_legacy import MorseScorerV3, load_legacy_scorer
from .retriever import BaseRetriever, ScoreConfig


class HybridMorseRetriever(BaseRetriever):
    def __init__(self, mkg: MKG, index: TripletIndex, clip,
                 mm_scorer: MorseScorer, text_scorer: MorseScorerV3,
                 cfg: Optional[ScoreConfig] = None):
        super().__init__(mkg, index, clip, cfg or ScoreConfig())
        self.mm = MorseRetriever(mkg, index, clip, mm_scorer)
        # text-only side uses the legacy 3-expert forward directly; no
        # separate retriever class needed, we score+topk here.
        self.text_scorer = text_scorer.eval()

    @classmethod
    def load(cls, mkg, index, clip, mm_ckpt: str, text_ckpt: str):
        mm_scorer = MorseScorer(index.e_txt.shape[1])
        state = torch.load(mm_ckpt, map_location="cpu")
        from .morse import MorseConfig
        mm_scorer = MorseScorer(state["dim"], MorseConfig(**state["cfg"]))
        mm_scorer.load_state_dict(state["model"])
        text_scorer = load_legacy_scorer(text_ckpt)
        return cls(mkg, index, clip, mm_scorer, text_scorer)

    @torch.no_grad()
    def _retrieve_text_only(self, queries: List[Query], cand_mask, depth: int,
                            batch_size: int, cand_chunk: int = 20000):
        idx = self.idx
        device = next(self.text_scorer.parameters()).device
        e_txt = idx.e_txt.to(device); e_img = idx.e_img.to(device)
        e_ctx = idx.e_ctx.to(device); e_rel = idx.e_rel.to(device)
        has_img = idx.has_img.to(device)
        C = e_txt.shape[0]
        mask_t = torch.tensor(cand_mask, dtype=torch.bool, device=device) \
            if cand_mask is not None else None
        out = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            qbar, delta_q, _, _ = self.embed_queries_with_delta(batch)
            qbar = qbar.to(device); delta_q = delta_q.to(device)
            B = qbar.shape[0]
            k = min(depth, C)
            best_scores = torch.full((B, k), float("-inf"), device=device)
            best_idx = torch.zeros((B, k), dtype=torch.long, device=device)
            for c0 in range(0, C, cand_chunk):
                c1 = min(c0 + cand_chunk, C)
                s = self.text_scorer(qbar, e_txt[c0:c1], e_img[c0:c1],
                                     e_ctx[c0:c1], e_rel[c0:c1],
                                     has_img[c0:c1], delta_q)
                if mask_t is not None:
                    s = s.masked_fill(~mask_t[c0:c1].unsqueeze(0), float("-inf"))
                kk = min(k, c1 - c0)
                cv, cp = torch.topk(s, kk, dim=1)
                ci = cp + c0
                cat_v = torch.cat([best_scores, cv], dim=1)
                cat_i = torch.cat([best_idx, ci], dim=1)
                best_scores, sel = torch.topk(cat_v, k, dim=1)
                best_idx = torch.gather(cat_i, 1, sel)
            out.extend(best_idx.tolist())
        return out

    def retrieve_ranked(self, queries: List[Query], cand_mask=None,
                        depth: int = 100, batch_size: int = 64):
        """Splits queries by has_visual_evidence, scores each group with
        its dedicated model, and reassembles results in the ORIGINAL
        query order so this is a drop-in replacement for any other
        retriever's retrieve_ranked."""
        mm_idx = [i for i, q in enumerate(queries) if q.has_visual_evidence]
        text_idx = [i for i, q in enumerate(queries) if not q.has_visual_evidence]

        results = [None] * len(queries)
        if mm_idx:
            mm_queries = [queries[i] for i in mm_idx]
            mm_out = self.mm.retrieve_ranked(mm_queries, cand_mask, depth, batch_size)
            for pos, i in enumerate(mm_idx):
                results[i] = mm_out[pos]
        if text_idx:
            text_queries = [queries[i] for i in text_idx]
            text_out = self._retrieve_text_only(text_queries, cand_mask, depth, batch_size)
            for pos, i in enumerate(text_idx):
                results[i] = text_out[pos]
        return results

    def retrieve(self, queries, cand_mask=None, topk=None, batch_size=64):
        return self.retrieve_ranked(queries, cand_mask,
                                    depth=topk or self.mm.cfg.K,
                                    batch_size=batch_size)
