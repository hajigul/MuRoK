"""Legacy 3-expert MORSE (lexical, visual, structural), kept ALONGSIDE the
current 5-expert morse.py so checkpoints trained with it (e.g.
morse_fb_v3.pt) can still be loaded. This is the architecture that,
empirically on FB15k-237-IMG, wins the text-only settings (S2/S3) but
loses the image-rich settings (S1/S4/S5) to the 5-expert version in
morse.py -- see morse/hybrid.py, which dispatches between the two per
query rather than picking one architecture for everything.

Do not add new features here; this module exists purely for checkpoint
compatibility and as one half of the hybrid retriever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MorseConfigV3:
    hidden: int = 64
    temperature: float = 1.0
    K: int = 5
    depth: int = 100


class MorseGateV3(nn.Module):
    """[qbar ; e_rel ; has_img ; delta_q ; s_txt ; s_img ; s_ctx] -> 3 expert logits."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        in_dim = 2 * dim + 1 + 1 + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, qbar, e_rel, has_img, delta_q, s_txt, s_img, s_ctx):
        if e_rel.dim() == 2:
            C = e_rel.shape[0]
            B = qbar.shape[0]
            q = qbar.unsqueeze(1).expand(B, C, -1)
            r = e_rel.unsqueeze(0).expand(B, C, -1)
            m = has_img.view(1, C, 1).expand(B, C, 1).float()
        else:
            B, C, _ = e_rel.shape
            q = qbar.unsqueeze(1).expand(B, C, -1)
            r = e_rel
            m = has_img.unsqueeze(-1).float()
        dq = delta_q.view(B, 1, 1).expand(B, C, 1).float()
        x = torch.cat([q, r, m, dq,
                       s_txt.unsqueeze(-1), s_img.unsqueeze(-1), s_ctx.unsqueeze(-1)],
                      dim=-1)
        return self.net(x)


class MorseScorerV3(nn.Module):
    def __init__(self, dim: int, cfg: Optional[MorseConfigV3] = None):
        super().__init__()
        self.cfg = cfg or MorseConfigV3()
        self.gate = MorseGateV3(dim, self.cfg.hidden)

    def forward(self, qbar, e_txt, e_img, e_ctx, e_rel, has_img, delta_q):
        if e_txt.dim() == 2:
            s_txt = qbar @ e_txt.T
            s_img = qbar @ e_img.T
            s_ctx = qbar @ e_ctx.T
            mask = has_img.view(1, -1)
        else:
            s_txt = (qbar.unsqueeze(1) * e_txt).sum(-1)
            s_img = (qbar.unsqueeze(1) * e_img).sum(-1)
            s_ctx = (qbar.unsqueeze(1) * e_ctx).sum(-1)
            mask = has_img
        mask_f = mask.float()
        s_img_masked = s_img * mask_f
        logits = self.gate(qbar, e_rel, has_img, delta_q, s_txt, s_img_masked, s_ctx)
        neg = torch.finfo(logits.dtype).min
        logits = logits.clone()
        logits[..., 1] = torch.where(mask.bool(), logits[..., 1],
                                     torch.full_like(logits[..., 1], neg))
        w = F.softmax(logits / self.cfg.temperature, dim=-1)
        experts = torch.stack([s_txt, s_img_masked, s_ctx], dim=-1)
        return (w * experts).sum(-1)


def load_legacy_scorer(ckpt_path: str) -> MorseScorerV3:
    state = torch.load(ckpt_path, map_location="cpu")
    scorer = MorseScorerV3(state["dim"], MorseConfigV3(**state["cfg"]))
    scorer.load_state_dict(state["model"])
    return scorer.eval()
