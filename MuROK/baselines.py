"""Additional baselines matching the benchmark's method families:

* CaptioningRetriever — caption-then-retrieve: every image (query and
  candidate) is replaced by a BLIP caption appended to the text, and
  retrieval is pure text matching with the frozen CLIP text encoder.
  Requires captions.json produced by scripts/build_captions.py.

* RerankingRetriever — two-stage retrieval in the spirit of EchoSight:
  stage one ranks candidates by image-image similarity for multimodal
  queries (text similarity as fallback for image-less candidates, and for
  text-only queries), then the top-N shortlist is reranked by the mean of
  text-text and image-image similarity.

Both reuse the frozen CLIP encoders and the cached TripletIndex, so every
method in the results table shares identical representations.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import torch

from .data import MKG, Query
from .index import TripletIndex
from .retriever import ScoreConfig, BaseRetriever


class CaptioningRetriever(BaseRetriever):
    def __init__(self, mkg: MKG, index: TripletIndex, clip, data_root: str,
                 cfg: Optional[ScoreConfig] = None):
        super().__init__(mkg, index, clip, cfg or ScoreConfig())
        cap_path = os.path.join(data_root, "captions.json")
        if not os.path.exists(cap_path):
            raise FileNotFoundError(
                f"{cap_path} not found - run scripts/build_captions.py first")
        with open(cap_path, encoding="utf-8") as f:
            self.captions = json.load(f)
        cache = os.path.join(data_root, "index_captioning.pt")
        if os.path.exists(cache):
            self.e_cap = torch.load(cache, map_location="cpu")
        else:
            texts = []
            for c in mkg.triplets:
                cap = self._cap(c.image)
                texts.append(f"{c.verbalize()} | {cap}" if cap else c.verbalize())
            print(f"[captioning] encoding {len(texts)} captioned candidates (cached after)")
            self.e_cap = clip.encode_text(texts)
            torch.save(self.e_cap, cache)

    def _cap(self, img_path):
        if not img_path:
            return None
        return self.captions.get(os.path.abspath(img_path))

    def retrieve_ranked(self, queries: List[Query], cand_mask=None,
                        depth: int = 100, batch_size: int = 64):
        mask_t = torch.tensor(cand_mask, dtype=torch.bool) if cand_mask is not None else None
        out = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            texts = []
            for q in batch:
                cap = self._cap(q.image) if (q.image and os.path.exists(q.image)) else None
                texts.append(f"{q.text} | {cap}" if cap else q.text)
            qe = self.clip.encode_text(texts)
            s = qe @ self.e_cap.T
            if mask_t is not None:
                s = s.masked_fill(~mask_t.unsqueeze(0), float("-inf"))
            k = min(depth, s.shape[1])
            out.extend(torch.topk(s, k, dim=1).indices.tolist())
        return out

    def retrieve(self, queries, cand_mask=None, topk=None, batch_size=64):
        return self.retrieve_ranked(queries, cand_mask,
                                    depth=topk or self.cfg.K, batch_size=batch_size)


class RerankingRetriever(BaseRetriever):
    def __init__(self, mkg: MKG, index: TripletIndex, clip,
                 cfg: Optional[ScoreConfig] = None, N: int = 200):
        super().__init__(mkg, index, clip, cfg or ScoreConfig())
        self.N = N

    def retrieve_ranked(self, queries: List[Query], cand_mask=None,
                        depth: int = 100, batch_size: int = 64):
        mask_t = torch.tensor(cand_mask, dtype=torch.bool) if cand_mask is not None else None
        has_img = self.idx.has_img
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            qt = self.clip.encode_text([q.text for q in batch])
            qi = torch.zeros_like(qt)
            q_has = torch.zeros(len(batch), dtype=torch.bool)
            pos = [j for j, q in enumerate(batch)
                   if q.image and os.path.exists(q.image)]
            if pos:
                emb = self.clip.encode_images([batch[j].image for j in pos])
                for r, j in enumerate(pos):
                    qi[j] = emb[r]
                    q_has[j] = True
            sim_t = qt @ self.idx.e_txt.T          # [B, T]
            sim_i = qi @ self.idx.e_img.T          # [B, T]
            for b in range(len(batch)):
                if q_has[b]:
                    s1 = torch.where(has_img, sim_i[b], sim_t[b])
                    s2 = torch.where(has_img, 0.5 * sim_t[b] + 0.5 * sim_i[b], sim_t[b])
                else:
                    s1 = sim_t[b]
                    s2 = sim_t[b]
                if mask_t is not None:
                    s1 = s1.masked_fill(~mask_t, float("-inf"))
                    s2 = s2.masked_fill(~mask_t, float("-inf"))
                N = min(self.N, s1.shape[0])
                short = torch.topk(s1, N).indices
                order = short[torch.argsort(s2[short], descending=True)].tolist()
                if depth > len(order):
                    seen = set(order)
                    extra = [x for x in torch.topk(
                        s1, min(depth + N, s1.shape[0])).indices.tolist()
                        if x not in seen]
                    order += extra
                results.append(order[:depth])
        return results

    def retrieve(self, queries, cand_mask=None, topk=None, batch_size=64):
        return self.retrieve_ranked(queries, cand_mask,
                                    depth=topk or self.cfg.K, batch_size=batch_size)


def make_extended_baseline(name: str, mkg: MKG, index: TripletIndex, clip,
                           data_root: str):
    name = name.lower()
    if name == "captioning":
        return CaptioningRetriever(mkg, index, clip, data_root)
    if name == "reranking":
        return RerankingRetriever(mkg, index, clip)
    raise ValueError(f"Unknown extended baseline {name}")
