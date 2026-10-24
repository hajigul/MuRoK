"""Caption every image referenced by a dataset (candidate triplet images and
query images) with BLIP-base, saving data_root/captions.json keyed by
absolute image path. Needed once per dataset before using the Captioning
baseline.

  python scripts/build_captions.py --data data/fb15k237img
  python scripts/build_captions.py --data data/bench_g

First run downloads Salesforce/blip-image-captioning-base (~1 GB) from
HuggingFace (no account/key needed). Roughly 8-15 images/sec on a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm

from morse.data import MKG, load_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    mkg = MKG.load(args.data)
    paths = set()
    for c in mkg.triplets:
        if c.image and os.path.exists(c.image):
            paths.add(os.path.abspath(c.image))
    for split in ["train", "val", "test"]:
        try:
            for q in load_queries(args.data, split):
                if q.image and os.path.exists(q.image):
                    paths.add(os.path.abspath(q.image))
        except FileNotFoundError:
            pass
    paths = sorted(paths)
    print(f"{len(paths)} unique images to caption")

    out_path = os.path.join(args.data, "captions.json")
    captions = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            captions = json.load(f)
        paths = [p for p in paths if p not in captions]
        print(f"resuming: {len(captions)} cached, {len(paths)} remaining")
    if not paths:
        print("nothing to do")
        return

    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name).to(device).eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), args.batch_size), desc="caption"):
            batch = paths[i: i + args.batch_size]
            imgs = []
            ok = []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    ok.append(p)
                except Exception:
                    captions[p] = ""
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            for p, o in zip(ok, out):
                captions[p] = processor.decode(o, skip_special_tokens=True).strip()
            if (i // args.batch_size) % 50 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(captions, f, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False)
    print(f"wrote {len(captions)} captions -> {out_path}")


if __name__ == "__main__":
    main()
