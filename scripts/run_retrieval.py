"""Run retrieval evaluation (Tables 3/4 style).

Examples:
  # toy smoke test, all settings, MORSE + Fusion baseline
  python scripts/run_retrieval.py --data data/toy --split test --settings S1 S2 S3 S4 S5

  # real benchmark subset
  python scripts/run_retrieval.py --data data/bench_g --split test --settings S1 S4 \
      --methods morse fusion text-only --save outputs/bench_g_retrieval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from morse.data import MKG, filter_setting, load_queries
from morse.encoders import FrozenCLIP
from morse.index import TripletIndex, build_index
from morse.metrics import evaluate_retrieval
from morse.retriever import ScoreConfig, BaseRetriever, make_baseline


def get_index(mkg: MKG, clip: FrozenCLIP, data_root: str, m: int, rebuild: bool) -> TripletIndex:
    cache = os.path.join(data_root, f"index_m{m}.pt")
    if os.path.exists(cache) and not rebuild:
        print(f"Loading cached index: {cache}")
        idx = TripletIndex.load(cache)
        if not idx.has_caption.any() and os.path.exists(os.path.join(data_root, "captions.json")):
            print("[warn] cached index has no caption embeddings but captions.json "
                  "exists -- rebuilding so the caption expert is populated "
                  "(pass --rebuild-index to silence this next time)")
            rebuild = True
        else:
            return idx
    print("Building index (offline embeddings, cached for reuse)...")
    idx = build_index(mkg, clip, m=m, data_root=data_root)
    idx.save(cache)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--settings", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--methods", nargs="+", default=["morse", "fusion", "text-only"])
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--save", default=None)
    # MORSE hyperparameters (paper defaults)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--kappa", type=int, default=8)
    ap.add_argument("--morse-ckpt", default=None,
                    help="checkpoint for --methods morse (and the "
                         "multimodal side of morse-hybrid)")
    ap.add_argument("--morse-text-ckpt", default=None,
                    help="checkpoint for the text-only side of "
                         "--methods morse-hybrid (a LEGACY 3-expert "
                         "checkpoint, e.g. morse_fb_v3.pt)")
    ap.add_argument("--lam", type=float, default=0.7)
    args = ap.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    mkg = MKG.load(args.data)
    print(f"MKG: {len(mkg)} triplets, {len(mkg.entities)} entities, {len(mkg.relations)} relations")
    queries = load_queries(args.data, args.split)
    print(f"Queries ({args.split}): {len(queries)} "
          f"({sum(q.is_multimodal for q in queries)} multimodal)")

    clip = FrozenCLIP()
    index = get_index(mkg, clip, args.data, args.m, args.rebuild_index)

    all_results = {}
    for setting in args.settings:
        qs, cand_mask = filter_setting(queries, mkg, setting)
        if not qs:
            print(f"[{setting}] no queries in this setting, skipping")
            continue
        print(f"\n===== Setting {setting}: {len(qs)} queries, "
              f"{sum(cand_mask)} candidate triplets =====")
        gold_sets = [set(mkg.tid2idx[t] for t in q.gold_triplets if t in mkg.tid2idx) for q in qs]

        for method in args.methods:
            if method == "morse":
                from morse.morse import MorseRetriever
                assert args.morse_ckpt, "--morse-ckpt required for morse"
                retriever = MorseRetriever.load(mkg, index, clip, args.morse_ckpt)
            elif method == "morse-hybrid":
                from morse.hybrid import HybridMorseRetriever
                assert args.morse_ckpt, "--morse-ckpt (multimodal side) required for morse-hybrid"
                assert args.morse_text_ckpt, "--morse-text-ckpt (text-only side) required for morse-hybrid"
                retriever = HybridMorseRetriever.load(mkg, index, clip,
                                                      args.morse_ckpt, args.morse_text_ckpt)
            elif method in ("captioning", "reranking"):
                from morse.baselines import make_extended_baseline
                retriever = make_extended_baseline(method, mkg, index, clip, args.data)
            else:
                retriever = make_baseline(method, mkg, index, clip)

            t0 = time.time()
            rankings = retriever.retrieve_ranked(qs, cand_mask, depth=args.depth)
            dt = time.time() - t0
            metrics = evaluate_retrieval(rankings, gold_sets)
            qps = len(qs) / dt if dt > 0 else float("inf")
            print(f"\n[{setting}] {method.upper()}  (QPS: {qps:.2f})")
            for k in (5, 10, 20, 50, 100):
                print(f"  NDCG@{k:<3} {metrics[f'NDCG@{k}']:6.2f}   "
                      f"P@{k:<3} {metrics[f'P@{k}']:6.2f}   "
                      f"R@{k:<3} {metrics[f'R@{k}']:6.2f}")
            all_results[f"{setting}/{method}"] = {**metrics, "QPS": qps}

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
