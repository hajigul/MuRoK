"""Inspect what the MORSE router learned: mean expert weights per relation,
across all 5 experts (text, visual, structural, caption, image-image).

  python scripts/inspect_router.py --data data/fb15k237img \
      --ckpt results/morse_fb_v4.pt --split val --top 20
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from collections import defaultdict
from morse.data import MKG, load_queries
from morse.encoders import FrozenCLIP
from morse.index import build_index
from morse.morse import MorseScorer, MorseConfig
from morse.retriever import BaseRetriever, ScoreConfig

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--m", type=int, default=8)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
mkg = MKG.load(args.data)
clip = FrozenCLIP(device=device)
idx = build_index(mkg, clip, m=args.m, data_root=args.data)
qs = load_queries(args.data, args.split)

state = torch.load(args.ckpt, map_location="cpu")
scorer = MorseScorer(state["dim"], MorseConfig(**state["cfg"]))
scorer.load_state_dict(state["model"]); scorer.eval()

fus = BaseRetriever(mkg, idx, clip, ScoreConfig(use_scv=False, use_rca=False,
                                               use_gcr=False, gamma=0.0))
qb, dq, qimg, _ = fus.embed_queries_with_delta(qs[:512])
print(f"queries with a resolvable image (delta_q=1): {int(dq.sum())}/{len(dq)}")

EXPERT_NAMES = ["lexical", "visual", "struct", "caption", "img-img"]
rel_w = defaultdict(lambda: [0.0] * (len(EXPERT_NAMES) + 1))  # +1 for count
with torch.no_grad():
    for i in range(0, qb.shape[0], 64):
        q = qb[i:i+64]
        d = dq[i:i+64]
        qi = qimg[i:i+64]
        s_txt = q @ idx.e_txt.T
        s_img = q @ idx.e_img.T
        s_ctx = q @ idx.e_ctx.T
        s_cap = q @ idx.e_cap.T
        s_ii = qi @ idx.e_img.T
        mask_img = idx.has_img.view(1, -1)
        mask_cap = idx.has_caption.view(1, -1)
        mask_ii = mask_img.float() * d.view(-1, 1)
        s_img_m = s_img * mask_img.float()
        s_cap_m = s_cap * mask_cap.float()
        s_ii_m = s_ii * mask_ii
        scores = torch.stack([s_txt, s_img_m, s_ctx, s_cap_m, s_ii_m], dim=-1)
        logits = scorer.gate(q, idx.e_rel, idx.has_img, d, scores)
        neg = torch.finfo(logits.dtype).min
        logits = logits.clone()
        logits[..., 1] = torch.where(mask_img.bool(), logits[..., 1],
                                     torch.full_like(logits[..., 1], neg))
        logits[..., 3] = torch.where(mask_cap.bool(), logits[..., 3],
                                     torch.full_like(logits[..., 3], neg))
        logits[..., 4] = torch.where(mask_ii.bool(), logits[..., 4],
                                     torch.full_like(logits[..., 4], neg))
        w = F.softmax(logits / scorer.cfg.temperature, dim=-1).mean(0)  # [C,5]
        for j, c in enumerate(mkg.triplets):
            acc = rel_w[c.r]
            for e in range(len(EXPERT_NAMES)):
                acc[e] += float(w[j, e])
            acc[-1] += 1

rows = []
for r, vals in rel_w.items():
    n = vals[-1]
    if n:
        rows.append((r, [v / n for v in vals[:-1]], n))
rows.sort(key=lambda x: -x[1][1])  # sort by visual-expert weight, descending

header = f"{'relation':<40}" + "".join(f"{nm:>9}" for nm in EXPERT_NAMES) + f"{'#cand':>8}"
print(f"\n{header}")
print("-" * len(header))
for r, w, n in rows[:args.top]:
    print(f"{r[:39]:<40}" + "".join(f"{v:9.3f}" for v in w) + f"{n:8d}")
print("\n(sorted by visual-expert weight, descending)")
