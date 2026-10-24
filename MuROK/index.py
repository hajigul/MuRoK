"""Offline indexing (Algorithm 1, line 1).

For every candidate triplet c we cache:
  e_txt[c] = fT(verb(c))
  e_img[c] = fI(v_h)              (zero vector + mask=False if no image)
  e_rel[c] = fT(r)
  e_ctx[c] = fT( ⊕_{c' in N_m(c)} verb(c') )   (SCV, Eq. 2)
  e_cap[c] = fT(verb(c) + " | " + BLIP-caption(v_h))   (caption view, optional)

Concatenate-then-encode: neighbors are joined into ONE context string and
encoded jointly (not averaged), per the paper.

e_cap is a 4th, optional candidate view: the same "caption-then-retrieve"
representation used by the Captioning baseline (morse/baselines.py). It
lets MORSE's router pick a caption-augmented TEXT-TEXT comparison instead
of being limited to e_img's cross-modal TEXT-IMAGE comparison, which is
weaker due to CLIP's text/image modality gap. Built only if
`data_root/captions.json` exists (run scripts/build_captions.py first);
if absent, e_cap falls back to e_txt (equivalent to giving the caption
expert zero extra information — safe no-op, not an error) so older
datasets/checkpoints without captions still work.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm

from .data import MKG
from .encoders import FrozenCLIP


@dataclass
class TripletIndex:
    e_txt: torch.Tensor     # [T, d]
    e_img: torch.Tensor     # [T, d] (zeros where has_img is False)
    e_rel: torch.Tensor     # [T, d]
    e_ctx: torch.Tensor     # [T, d]
    e_cap: torch.Tensor     # [T, d] (== e_txt where no caption available)
    has_img: torch.Tensor   # [T] bool
    has_caption: torch.Tensor  # [T] bool

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "e_txt": self.e_txt,
                "e_img": self.e_img,
                "e_rel": self.e_rel,
                "e_ctx": self.e_ctx,
                "e_cap": self.e_cap,
                "has_img": self.has_img,
                "has_caption": self.has_caption,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TripletIndex":
        d = torch.load(path, map_location="cpu")
        if "e_cap" not in d:  # backward-compat with pre-caption-expert caches
            d["e_cap"] = d["e_txt"].clone()
            d["has_caption"] = torch.zeros(d["e_txt"].shape[0], dtype=torch.bool)
        return cls(**d)


def build_index(mkg: MKG, clip: FrozenCLIP, m: int = 8, batch_size: int = 256,
                data_root: Optional[str] = None) -> TripletIndex:
    T = len(mkg)
    d = clip.dim

    # --- triplet text embeddings ------------------------------------------------
    verb = [c.verbalize() for c in mkg.triplets]
    e_txt = clip.encode_text(verb, batch_size)

    # --- relation embeddings (encode unique relations once) ----------------------
    rel_names = sorted({c.r_name for c in mkg.triplets})
    rel_emb = clip.encode_text(rel_names, batch_size)
    rel2row = {r: i for i, r in enumerate(rel_names)}
    e_rel = torch.stack([rel_emb[rel2row[c.r_name]] for c in mkg.triplets])

    # --- image embeddings ---------------------------------------------------------
    e_img = torch.zeros(T, d)
    has_img = torch.zeros(T, dtype=torch.bool)
    img_idx = [i for i, c in enumerate(mkg.triplets) if c.image_ok]
    if img_idx:
        # encode each unique image once
        uniq = sorted({mkg.triplets[i].image for i in img_idx})
        uniq_emb = clip.encode_images(list(uniq))
        path2row = {p: j for j, p in enumerate(uniq)}
        for i in tqdm(img_idx, desc="assign image embeddings", leave=False):
            e_img[i] = uniq_emb[path2row[mkg.triplets[i].image]]
            has_img[i] = True

    # --- SCV context embeddings (Eq. 2) ------------------------------------------
    ctx_strings = []
    for i in range(T):
        nbrs = mkg.neighbors.get(i, [])[:m]
        if nbrs:
            ctx_strings.append(" ; ".join(mkg.triplets[j].verbalize() for j in nbrs))
        else:
            # isolated candidate: fall back to its own verbalization
            ctx_strings.append(mkg.triplets[i].verbalize())
    e_ctx = clip.encode_text(ctx_strings, batch_size)

    # --- caption embeddings (optional 4th expert) ---------------------------------
    e_cap = e_txt.clone()
    has_caption = torch.zeros(T, dtype=torch.bool)
    cap_path = os.path.join(data_root, "captions.json") if data_root else None
    if cap_path and os.path.exists(cap_path):
        with open(cap_path, encoding="utf-8") as f:
            captions = json.load(f)
        cap_texts = []
        cap_idx = []
        for i, c in enumerate(mkg.triplets):
            if has_img[i]:
                cap = captions.get(os.path.abspath(c.image))
                if cap:
                    cap_texts.append(f"{c.verbalize()} | {cap}")
                    cap_idx.append(i)
        if cap_texts:
            print(f"[index] encoding {len(cap_texts)} caption-augmented candidates")
            cap_emb = clip.encode_text(cap_texts, batch_size)
            for row, i in enumerate(cap_idx):
                e_cap[i] = cap_emb[row]
                has_caption[i] = True

    return TripletIndex(e_txt=e_txt, e_img=e_img, e_rel=e_rel, e_ctx=e_ctx,
                        e_cap=e_cap, has_img=has_img, has_caption=has_caption)
