"""MORSE: Mixture Of Retrieval Score Experts.

A learned alternative for multimodal KG triplet retrieval. The frozen CLIP
encoders and cached TripletIndex are reused unchanged; MORSE adds a small
gating network that outputs, per query-candidate pair, mixture weights over
FIVE scoring experts:

    expert 1: text similarity        <qbar, e_txt>
    expert 2: visual similarity      <qbar, e_img>   (cross-modal; masked when no image)
    expert 3: structural similarity  <qbar, e_ctx>   (SCV neighborhood view)
    expert 4: caption similarity     <qbar, e_cap>   (caption-augmented TEXT-TEXT view; masked when no caption)
    expert 5: image-image similarity <q_img, e_img>  (pure image-image view; masked unless BOTH query and candidate have an image)

Experts 4 and 5 exist to close a specific, diagnosed gap: expert 2 compares
the additively-fused query embedding q̄ (Eq. 1) against a candidate image,
which is a CROSS-MODAL (text/image) comparison. CLIP's embedding space has
a well-documented "modality gap" that makes cross-modal similarity noisier
than same-modality similarity. Two non-learned baselines exploit this:
Captioning turns images into text and does TEXT-TEXT matching; two-stage
reranking compares query and candidate images directly, i.e. IMAGE-IMAGE
matching. Both outperformed the original 3-expert MORSE on FB15k-237-IMG's
image-rich settings. Experts 4 and 5 give the SAME two same-modality
comparisons to the learned router, so it can select them per-relation
instead of being capped by cross-modal similarity alone.

The gate g(.) is an MLP over
    [qbar ; e_rel ; has_img ; delta_q ; s_txt ; s_img ; s_ctx ; s_cap ; s_ii]
so routing is relation-specific and modality-aware on both the candidate
side (has_img) and the query side (delta_q), and can see all five experts'
own raw similarities rather than only re-deriving confidence from
embeddings. Score:

    s(q,c) = sum_k  w_k(q,c) * expert_k(q,c),   w = softmax(g(.)/T)

with any expert requiring an image masked to -inf before the softmax when
that image doesn't exist (graceful degradation by construction, mirroring
Proposition P1: image-free candidates fall back to the text/structural/
caption(=text) experts only, and image-image additionally requires the
QUERY to have an image too).

Trained with a listwise InfoNCE ranking loss over gold triplets against
in-batch and fusion-mined hard negatives. The encoders and index are never
updated; only the gate g_theta is trainable.

CHANGE LOG (checkpoints from earlier gate shapes are NOT state-dict
compatible and must be retrained):
  - v2: added delta_q (query-side image indicator) and raw expert scores
    (s_txt, s_img, s_ctx) as gate inputs, instead of embeddings alone.
  - v3 (this version): added the caption expert (s_cap) and the
    image-image expert (s_ii) described above, motivated by MORSE losing
    to Captioning/Reranking baselines on FB15k-237-IMG's image-rich
    settings (All/All, Multimodal/Multimodal, Multimodal/All) despite
    beating them on text-only settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MKG, Query
from .index import TripletIndex
from .retriever import BaseRetriever, ScoreConfig

N_EXPERTS = 5  # text, visual, structural, caption, image-image


@dataclass
class MorseConfig:
    hidden: int = 64
    temperature: float = 1.0     # softmax temperature of the gate (post-anneal value)
    K: int = 5
    depth: int = 100
    # --- expert ablation flags -------------------------------------------
    # When True, the corresponding expert is FORCE-MASKED to -inf at every
    # step of training and inference, regardless of modality availability,
    # so the router never sees or can route to it. This is distinct from
    # the data-driven masks in forward() (which only fire when an image/
    # caption is actually missing) -- these are permanent, architectural
    # ablations for the expert-ablation study. Saved/restored with the
    # checkpoint so an ablated model can't be accidentally evaluated as if
    # it still had that expert.
    disable_caption: bool = False
    disable_image_image: bool = False
    disable_visual: bool = False


class MorseGate(nn.Module):
    """MLP gate: [qbar ; e_rel ; has_img ; delta_q ; s_txt ; s_img ; s_ctx ; s_cap ; s_ii] -> 5 expert logits."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        in_dim = 2 * dim + 1 + 1 + N_EXPERTS  # qbar, e_rel, has_img, delta_q, 5 raw expert scores
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_EXPERTS),
        )

    def forward(self, qbar: torch.Tensor, e_rel: torch.Tensor,
                has_img: torch.Tensor, delta_q: torch.Tensor,
                scores: torch.Tensor) -> torch.Tensor:
        """qbar [B,d] broadcast against candidates [B,C,d] or [C,d].
        delta_q: [B] (0/1 float). scores: [B,C,N_EXPERTS], the raw expert
        scores already computed by the caller (image-dependent experts
        should already be zeroed where masked; masking of LOGITS for the
        softmax happens in MorseScorer, not here)."""
        if e_rel.dim() == 2:                      # [C,d] shared across batch
            C = e_rel.shape[0]
            B = qbar.shape[0]
            q = qbar.unsqueeze(1).expand(B, C, -1)
            r = e_rel.unsqueeze(0).expand(B, C, -1)
            m = has_img.view(1, C, 1).expand(B, C, 1).float()
        else:                                     # [B,C,d]
            B, C, _ = e_rel.shape
            q = qbar.unsqueeze(1).expand(B, C, -1)
            r = e_rel
            m = has_img.unsqueeze(-1).float()
        dq = delta_q.view(B, 1, 1).expand(B, C, 1).float()
        x = torch.cat([q, r, m, dq, scores], dim=-1)
        return self.net(x)                        # [B,C,N_EXPERTS]


class MorseScorer(nn.Module):
    def __init__(self, dim: int, cfg: Optional[MorseConfig] = None):
        super().__init__()
        self.cfg = cfg or MorseConfig()
        self.gate = MorseGate(dim, self.cfg.hidden)

    def forward(self, qbar, e_txt, e_img, e_ctx, e_cap, e_rel, has_img,
                has_caption, delta_q, q_img):
        """All candidate tensors [C,d] (shared) or [B,C,d]; returns [B,C].
        delta_q: [B] float, 1.0 iff the query's image was actually encoded.
        q_img: [B,d] raw (un-fused) query image embedding, zero rows where
        delta_q==0 (see BaseRetriever.embed_queries_with_delta)."""
        shared = e_txt.dim() == 2
        if shared:
            s_txt = qbar @ e_txt.T                # [B,C]
            s_img = qbar @ e_img.T
            s_ctx = qbar @ e_ctx.T
            s_cap = qbar @ e_cap.T
            s_ii = q_img @ e_img.T                # pure image-image, [B,C]
            mask_img = has_img.view(1, -1)
            mask_cap = has_caption.view(1, -1)
        else:
            s_txt = (qbar.unsqueeze(1) * e_txt).sum(-1)
            s_img = (qbar.unsqueeze(1) * e_img).sum(-1)
            s_ctx = (qbar.unsqueeze(1) * e_ctx).sum(-1)
            s_cap = (qbar.unsqueeze(1) * e_cap).sum(-1)
            s_ii = (q_img.unsqueeze(1) * e_img).sum(-1)
            mask_img = has_img
            mask_cap = has_caption

        mask_img_f = mask_img.float()
        mask_cap_f = mask_cap.float()
        mask_dq = delta_q.view(-1, 1)              # [B,1], broadcasts over C
        mask_ii = mask_img_f * mask_dq             # need BOTH candidate and query image

        # expert-ablation overrides: force a permanently-off mask
        # regardless of modality availability (see MorseConfig docstring)
        if self.cfg.disable_visual:
            mask_img_f = torch.zeros_like(mask_img_f)
        if self.cfg.disable_caption:
            mask_cap_f = torch.zeros_like(mask_cap_f)
        if self.cfg.disable_image_image:
            mask_ii = torch.zeros_like(mask_ii)

        s_img_m = s_img * mask_img_f
        s_cap_m = s_cap * mask_cap_f
        s_ii_m = s_ii * mask_ii

        scores = torch.stack([s_txt, s_img_m, s_ctx, s_cap_m, s_ii_m], dim=-1)  # [B,C,5]
        logits = self.gate(qbar, e_rel, has_img, delta_q, scores)              # [B,C,5]

        neg = torch.finfo(logits.dtype).min
        logits = logits.clone()
        logits[..., 1] = torch.where(mask_img_f.bool(), logits[..., 1],
                                     torch.full_like(logits[..., 1], neg))
        logits[..., 3] = torch.where(mask_cap_f.bool(), logits[..., 3],
                                     torch.full_like(logits[..., 3], neg))
        logits[..., 4] = torch.where(mask_ii.bool(), logits[..., 4],
                                     torch.full_like(logits[..., 4], neg))

        w = F.softmax(logits / self.cfg.temperature, dim=-1)  # [B,C,5]
        return (w * scores).sum(-1)                # [B,C]


class MorseRetriever(BaseRetriever):
    """Drop-in retriever: same interface as BaseRetriever/baselines."""

    def __init__(self, mkg: MKG, index: TripletIndex, clip, scorer: MorseScorer,
                 cfg: Optional[ScoreConfig] = None):
        super().__init__(mkg, index, clip, cfg or ScoreConfig())
        self.scorer = scorer.eval()

    @classmethod
    def load(cls, mkg, index, clip, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        scorer = MorseScorer(state["dim"], MorseConfig(**state["cfg"]))
        scorer.load_state_dict(state["model"])
        return cls(mkg, index, clip, scorer)

    @torch.no_grad()
    def retrieve_ranked(self, queries: List[Query], cand_mask=None,
                        depth: int = 100, batch_size: int = 64,
                        cand_chunk: int = 20000):
        """Memory-safe scoring.

        Queries are processed in batches of `batch_size`; within each query
        batch the candidate axis is streamed in chunks of `cand_chunk` so
        the full [B, C, 5] gate tensor is never materialized for very large
        indexes (e.g. FB15k-237-IMG with ~3e5 candidates). Top-`depth`
        indices are kept incrementally per chunk.
        """
        idx = self.idx
        device = next(self.scorer.parameters()).device
        e_txt = idx.e_txt.to(device)
        e_img = idx.e_img.to(device)
        e_ctx = idx.e_ctx.to(device)
        e_cap = idx.e_cap.to(device)
        e_rel = idx.e_rel.to(device)
        has_img = idx.has_img.to(device)
        has_caption = idx.has_caption.to(device)
        C = e_txt.shape[0]
        mask_t = torch.tensor(cand_mask, dtype=torch.bool, device=device) \
            if cand_mask is not None else None

        out = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            qbar, delta_q, q_img, _ = self.embed_queries_with_delta(batch)
            qbar = qbar.to(device)
            delta_q = delta_q.to(device)
            q_img = q_img.to(device)
            B = qbar.shape[0]
            k = min(depth, C)
            best_scores = torch.full((B, k), float("-inf"), device=device)
            best_idx = torch.zeros((B, k), dtype=torch.long, device=device)
            for c0 in range(0, C, cand_chunk):
                c1 = min(c0 + cand_chunk, C)
                s = self.scorer(qbar, e_txt[c0:c1], e_img[c0:c1],
                                e_ctx[c0:c1], e_cap[c0:c1], e_rel[c0:c1],
                                has_img[c0:c1], has_caption[c0:c1],
                                delta_q, q_img)
                if mask_t is not None:
                    s = s.masked_fill(~mask_t[c0:c1].unsqueeze(0),
                                      float("-inf"))
                kk = min(k, c1 - c0)
                chunk_val, chunk_pos = torch.topk(s, kk, dim=1)
                chunk_idx = chunk_pos + c0
                cat_val = torch.cat([best_scores, chunk_val], dim=1)
                cat_idx = torch.cat([best_idx, chunk_idx], dim=1)
                best_scores, sel = torch.topk(cat_val, k, dim=1)
                best_idx = torch.gather(cat_idx, 1, sel)
                del s
            out.extend(best_idx.tolist())
        return out

    def retrieve(self, queries, cand_mask=None, topk=None, batch_size=64):
        return self.retrieve_ranked(queries, cand_mask,
                                    depth=topk or self.cfg.K,
                                    batch_size=batch_size)


def save_checkpoint(scorer: MorseScorer, dim: int, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": scorer.state_dict(), "dim": dim,
                "cfg": {"hidden": scorer.cfg.hidden,
                        "temperature": scorer.cfg.temperature,
                        "disable_caption": scorer.cfg.disable_caption,
                        "disable_image_image": scorer.cfg.disable_image_image,
                        "disable_visual": scorer.cfg.disable_visual}}, path)
