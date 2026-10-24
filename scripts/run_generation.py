"""End-to-end generation evaluation (Table 5 style): retrieve top-K triplets
with MORSE (or a baseline), condition a frozen local generator, score
EM / F1 / Contains@1 / BLEU-1.

Generator: Qwen2.5-VL-7B-Instruct (multimodal, ~16 GB bf16 on your 24 GB GPU),
or Qwen2.5-7B-Instruct with --text-only. Add --load-4bit if you hit OOM.

Examples:
  python scripts/run_generation.py --data data/toy --split test --setting S4 --method morse --limit 30 --text-only
  python scripts/run_generation.py --data data/bench_m --split test --setting S4 --method morse --save outputs/bench_m_gen_s4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from morse.data import MKG, filter_setting, load_queries
from morse.encoders import FrozenCLIP
from morse.generator import (QwenTextGenerator, QwenVLGenerator,
                            evaluate_generation, format_evidence)
from morse.index import TripletIndex, build_index
from morse.retriever import ScoreConfig, BaseRetriever, make_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--setting", default="S1")
    ap.add_argument("--method", default="morse", choices=["morse", "fusion", "text-only"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="cap #queries (debugging)")
    ap.add_argument("--text-only", action="store_true", help="use text-only LLM generator")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit quantized generator")
    ap.add_argument("--with-evidence-images", action="store_true",
                    help="also pass retrieved triplet images to the VLM")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    mkg = MKG.load(args.data)
    queries = load_queries(args.data, args.split)
    qs, cand_mask = filter_setting(queries, mkg, args.setting)
    if args.limit:
        qs = qs[: args.limit]
    print(f"{len(qs)} queries | setting {args.setting} | method {args.method}")

    # ---- retrieval ----
    clip = FrozenCLIP()
    cache = os.path.join(args.data, "index_m8.pt")
    index = TripletIndex.load(cache) if os.path.exists(cache) else build_index(mkg, clip, m=8, data_root=args.data)
    if not os.path.exists(cache):
        index.save(cache)
    if args.method == "morse":
        retriever = BaseRetriever(mkg, index, clip, ScoreConfig(K=args.K))
    else:
        retriever = make_baseline(args.method, mkg, index, clip)
    rankings = retriever.retrieve(qs, cand_mask, topk=args.K)

    # ---- free CLIP GPU memory before loading the generator ----
    import gc, torch
    del retriever, index
    clip.model.to("cpu"); del clip
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- generation ----
    print("Loading generator (first run downloads weights from HuggingFace)...")
    gen = QwenTextGenerator(args.load_4bit) if args.text_only else QwenVLGenerator(args.load_4bit)

    preds, gold_lists, records = [], [], []
    for q, top in tqdm(list(zip(qs, rankings)), desc="generate"):
        evidence = format_evidence(mkg, top)
        ev_imgs = None
        if args.with_evidence_images and not args.text_only:
            ev_imgs = [mkg.triplets[i].image for i in top if mkg.triplets[i].image][:2]
        pred = gen.answer(q, evidence, ev_imgs)
        preds.append(pred)
        gold_lists.append(q.answers)
        records.append({"qid": q.qid, "question": q.question,
                        "pred": pred, "answers": q.answers,
                        "retrieved": [mkg.triplets[i].tid for i in top]})

    metrics = evaluate_generation(preds, gold_lists)
    print(f"\n[{args.setting}] {args.method.upper()} generation:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:6.2f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({"metrics": metrics, "records": records}, f, indent=2)
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
