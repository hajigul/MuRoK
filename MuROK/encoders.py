"""Frozen CLIP ViT-B/32 encoders (Radford et al. 2021), matching the
benchmark's backbone. All embeddings are L2-normalized so inner products
are cosine similarities in [-1, 1].
"""
from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-base-patch32"


class FrozenCLIP:
    def __init__(self, device: Optional[str] = None, model_name: str = CLIP_NAME):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        """fT(.). CLIP truncates to its 77-token context, which implements the
        SCV truncation-to-max-context rule."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode_images(self, paths: List[str], batch_size: int = 128) -> torch.Tensor:
        """fI(.)."""
        out = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
        return torch.cat(out, dim=0)

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim
