"""MORSE scoring (Algorithm 1) and baseline retrievers.

Equations (paper numbering):
  (1)  q̄ = norm( fT(q_txt) + 1[q_img] fI(q_img) )
  (3)  α(q,c) = σ(β ⟨q̄, e_r⟩) · 1[v_h]
  (4)  s_RCA  = (1-α) a ⟨q̄, e_txt⟩ + α b ⟨q̄, e_img⟩
  (5)  s1     = s_RCA + γ ⟨q̄, e_ctx⟩
  (6)  cons(q,c) = mean_{c' in N_κ(c)} s1(q,c')
  (7)  s2     = λ s1 + (1-λ) cons          (s2 = s1 for isolated candidates)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .data import MKG, Query
from .encoders import FrozenCLIP
from .index import TripletIndex


@dataclass
class ScoreConfig:
    # validation-selected hyperparameters from the paper
    m: int = 8          # SCV neighborhood size (used at indexing time)
    gamma: float = 0.25  # SCV context weight γ
    beta: float = 5.0    # RCA gate temperature β
    a: float = 0.5       # text base weight
    b: float = 0.5       # image base weight
    N: int = 200         # GCR shortlist size
    kappa: int = 8       # GCR neighborhood size κ
    K: int = 5           # retrieved triplets passed to generator
    lam: float = 0.7     # GCR interpolation λ
    # ablation switches
    use_scv: bool = True
    use_rca: bool = True
    use_gcr: bool = True


class BaseRetriever:
    def __init__(self, mkg: MKG, index: TripletIndex, clip: FrozenCLIP, cfg: ScoreConfig):
        self.mkg = mkg
        self.idx = index
        self.clip = clip
        self.cfg = cfg
        # precompute padded neighbor tensor for vectorized GCR
        kappa = cfg.kappa
        T = len(mkg)
        nbr = torch.full((T, kappa), -1, dtype=torch.long)
        for i in range(T):
            js = mkg.neighbors.get(i, [])[:kappa]
            for k, j in enumerate(js):
                nbr[i, k] = j
        self.nbr = nbr  # [T, κ], -1 = padding

    # ------------------------------------------------------------------ queries
    def embed_query(self, q: Query) -> torch.Tensor:
        """Eq. 1 — additive fusion, then renormalize."""
        e = self.clip.encode_text([q.text])[0]
        if q.image is not None:
            e = e + self.clip.encode_images([q.image])[0]
        return torch.nn.functional.normalize(e, dim=-1)

    def embed_queries(self, queries: List[Query]) -> torch.Tensor:
        qbar, _, _, _ = self.embed_queries_with_delta(queries)
        return qbar

    def embed_queries_with_delta(self, queries: List[Query]):
        """Same as embed_queries, but also returns:
          delta_q: [B] float, 1.0 iff the query's image actually resolved on
            disk and was encoded, 0.0 otherwise. Ground truth for whether q̄
            carries visual evidence — `Query.is_multimodal` alone is NOT
            sufficient, since it reflects benchmark pool membership and can
            be True even when the file is missing.
          q_img: [B, d] the RAW (un-fused, not additively combined with
            text) query image embedding, zero where delta_q==0. Kept
            separate from q̄ so a pure image-image expert can be computed
            without CLIP's text/image modality gap contaminating it via
            the additive-fusion step of Eq. 1.
          q_text: [B, d] the query embedding as it WOULD be if the query
            had no image at all (pure text, normalized). For a query that
            does have an image, q_text != qbar. Used for modality dropout
            during training: swapping qbar -> q_text lets us simulate a
            genuinely text-only query from a multimodal one, rather than
            merely flipping the delta_q flag while leaving image
            information baked into qbar via additive fusion.
        """
        et = self.clip.encode_text([q.text for q in queries])
        q_text = torch.nn.functional.normalize(et, dim=-1)
        img_pos = [i for i, q in enumerate(queries) if q.image_ok]
        q_img = torch.zeros_like(et)
        fused = et.clone()
        if img_pos:
            ei = self.clip.encode_images([queries[i].image for i in img_pos])
            for row, i in enumerate(img_pos):
                fused[i] = fused[i] + ei[row]
                q_img[i] = ei[row]
        qbar = torch.nn.functional.normalize(fused, dim=-1)
        delta_q = torch.zeros(len(queries), dtype=torch.float32)
        delta_q[img_pos] = 1.0
        return qbar, delta_q, q_img, q_text

    # ------------------------------------------------------------------ scoring
    def stage1(self, qbar: torch.Tensor, cand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """s1 over ALL triplets for a batch of queries. qbar: [B, d] -> [B, T]."""
        cfg = self.cfg
        sim_txt = qbar @ self.idx.e_txt.T                       # [B, T]
        sim_img = qbar @ self.idx.e_img.T                       # [B, T]
        sim_rel = qbar @ self.idx.e_rel.T                       # [B, T]
        sim_ctx = qbar @ self.idx.e_ctx.T                       # [B, T]
        has_img = self.idx.has_img.float().unsqueeze(0)         # [1, T]

        if cfg.use_rca:
            alpha = torch.sigmoid(cfg.beta * sim_rel) * has_img  # Eq. 3
        else:
            # static late fusion: β→0 limit (α = 1/2 on image-bearing candidates)
            alpha = 0.5 * has_img
        s = (1 - alpha) * cfg.a * sim_txt + alpha * cfg.b * sim_img  # Eq. 4
        if cfg.use_scv:
            s = s + cfg.gamma * sim_ctx                              # Eq. 5
        if cand_mask is not None:
            s = s.masked_fill(~cand_mask.unsqueeze(0), float("-inf"))
        return s

    def gcr(self, s1: torch.Tensor, cand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rerank the stage-1 top-N by neighborhood consistency (Eqs. 6-7).

        s1: [B, T]. Returns s2: [B, T] where entries outside the shortlist
        keep their s1 value scaled to remain below any shortlist score is NOT
        needed — we only ever take top-K from the shortlist, so we return s2
        equal to s1 outside the shortlist and reranked inside it.
        """
        cfg = self.cfg
        B, T = s1.shape
        s2 = s1.clone()
        N = min(cfg.N, T)
        top = torch.topk(s1, N, dim=1).indices                  # [B, N]

        nbr = self.nbr                                          # [T, κ]
        pad = nbr < 0
        nbr_safe = nbr.clamp(min=0)

        for b in range(B):
            sel = top[b]                                        # [N]
            js = nbr_safe[sel]                                  # [N, κ]
            valid = ~pad[sel]                                   # [N, κ]
            if cand_mask is not None:
                valid = valid & cand_mask[js]
            neigh_scores = s1[b][js]                            # [N, κ]
            neigh_scores = neigh_scores.masked_fill(~valid, 0.0)
            cnt = valid.sum(dim=1)                              # [N]
            cons = torch.where(
                cnt > 0,
                neigh_scores.sum(dim=1) / cnt.clamp(min=1),
                s1[b][sel],                                     # isolated: s2 = s1
            )
            s2[b, sel] = torch.where(
                cnt > 0,
                cfg.lam * s1[b][sel] + (1 - cfg.lam) * cons,     # Eq. 7
                s1[b][sel],
            )
        return s2

    def retrieve(
        self,
        queries: List[Query],
        cand_mask: Optional[List[bool]] = None,
        topk: Optional[int] = None,
        batch_size: int = 64,
    ) -> List[List[int]]:
        """Full MORSE pipeline. Returns top-K triplet indices per query."""
        K = topk or self.cfg.K
        mask_t = torch.tensor(cand_mask, dtype=torch.bool) if cand_mask is not None else None
        results: List[List[int]] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            qbar = self.embed_queries(batch)
            s1 = self.stage1(qbar, mask_t)
            s = self.gcr(s1, mask_t) if self.cfg.use_gcr else s1
            top = torch.topk(s, K, dim=1).indices
            results.extend(top.tolist())
        return results

    def retrieve_ranked(
        self,
        queries: List[Query],
        cand_mask: Optional[List[bool]] = None,
        depth: int = 100,
        batch_size: int = 64,
    ) -> List[List[int]]:
        """Top-`depth` ranking for metric computation (NDCG/P/R up to @100)."""
        return self.retrieve(queries, cand_mask, topk=depth, batch_size=batch_size)


# ------------------------------------------------------------------- baselines
def make_baseline(name: str, mkg: MKG, index: TripletIndex, clip: FrozenCLIP) -> BaseRetriever:
    """Baselines expressed as points in MORSE's hyperparameter space (P2)."""
    name = name.lower()
    if name == "fusion":
        cfg = ScoreConfig(use_scv=False, use_rca=False, use_gcr=False, gamma=0.0)
    elif name == "text-only":
        cfg = ScoreConfig(use_scv=False, use_rca=False, use_gcr=False, gamma=0.0, b=0.0)
    elif name == "morse":
        cfg = ScoreConfig()
    elif name == "random":
        import random as _rnd

        class _RandomRetriever(BaseRetriever):
            def retrieve_ranked(self, queries, cand_mask=None, depth=100, batch_size=64):
                if cand_mask is None:
                    pool = list(range(len(self.mkg)))
                else:
                    pool = [i for i, ok in enumerate(cand_mask) if ok]
                rng = _rnd.Random(0)
                return [rng.sample(pool, min(depth, len(pool))) for _ in queries]

            def retrieve(self, queries, cand_mask=None, topk=None, batch_size=64):
                return self.retrieve_ranked(queries, cand_mask, depth=topk or self.cfg.K)

        return _RandomRetriever(mkg, index, clip, ScoreConfig())
    else:
        raise ValueError(f"Unknown method {name}")
    return BaseRetriever(mkg, index, clip, cfg)
