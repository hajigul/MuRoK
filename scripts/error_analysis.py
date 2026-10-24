"""Comprehensive error analysis and qualitative analysis: MORSE vs Fusion.

For every query it computes, under both methods, the rank of the first
gold triplet, hit@K, and NDCG@K, then produces:

1. Outcome matrix: both-correct / MORSE-only (wins) / Fusion-only (losses) /
   both-wrong, at hit@K.
2. Breakdowns of MORSE's hit@K and wins/losses by:
   - query modality (text vs multimodal)
   - gold relation type (top relations)
   - gold-triplet modality (image-bearing vs text-only gold)
   - number of gold triplets per query (1 vs 2+)
3. Rank-movement stats: how far gold moved up/down under MORSE vs Fusion.
4. Qualitative examples: the top wins and losses with the query, its gold
   triplet(s), both methods' top-5 retrieved triplets, and the RCA gate
   value alpha on the gold candidate — ready to paste into a case-study
   section.

Outputs a markdown report + JSON. Example:

  python scripts/error_analysis.py --data data/bench_g --split test --setting S4 \
      --beta 5 --gamma 0 --lam 1.0 --out outputs/error_analysis_g_s4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from morse.data import MKG, filter_setting, load_queries
from morse.encoders import FrozenCLIP
from morse.index import TripletIndex, build_index
from morse.metrics import ndcg_at_k
from morse.retriever import ScoreConfig, BaseRetriever, make_baseline


def first_gold_rank(ranking, gold):
    for r, idx in enumerate(ranking):
        if idx in gold:
            return r + 1  # 1-based
    return None


def fmt_triplet(mkg, i):
    c = mkg.triplets[i]
    tag = "[img]" if c.image else "     "
    return f"{tag} ({c.h_name} | {c.r_name} | {c.t_name})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--setting", default="S4")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--n-examples", type=int, default=10, help="qualitative wins/losses to dump")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/error_analysis")
    args = ap.parse_args()

    mkg = MKG.load(args.data)
    queries = load_queries(args.data, args.split)
    qs, cand_mask = filter_setting(queries, mkg, args.setting)
    print(f"{len(qs)} queries | setting {args.setting} | K={args.K}")

    clip = FrozenCLIP()
    cache = os.path.join(args.data, "index_m8.pt")
    index = TripletIndex.load(cache) if os.path.exists(cache) else build_index(mkg, clip, m=8, data_root=args.data)

    morse = BaseRetriever(mkg, index, clip,
                         ScoreConfig(beta=args.beta, gamma=args.gamma, lam=args.lam,
                                    use_scv=args.gamma > 0, use_gcr=args.lam < 1.0))
    fusion = make_baseline("fusion", mkg, index, clip)

    print("Retrieving with MORSE...")
    r_morse = morse.retrieve_ranked(qs, cand_mask, depth=args.depth)
    print("Retrieving with Fusion...")
    r_fus = fusion.retrieve_ranked(qs, cand_mask, depth=args.depth)

    # per-query records ------------------------------------------------------
    K = args.K
    records = []
    for q, rg, rf in zip(qs, r_morse, r_fus):
        gold = set(mkg.tid2idx[t] for t in q.gold_triplets if t in mkg.tid2idx)
        if not gold:
            continue
        rank_g, rank_f = first_gold_rank(rg, gold), first_gold_rank(rf, gold)
        hit_g, hit_f = (rank_g or 10**9) <= K, (rank_f or 10**9) <= K
        # gold-side properties
        gold_rels = [mkg.triplets[i].r_name for i in gold]
        gold_has_img = any(mkg.triplets[i].image for i in gold)
        # RCA gate on the best-ranked gold candidate under MORSE
        gi = min(gold, key=lambda i: rg.index(i) if i in rg else 10**9)
        qbar = morse.embed_queries([q])
        alpha = float(torch.sigmoid(args.beta * (qbar @ index.e_rel[gi].unsqueeze(1))).item()) \
            if bool(index.has_img[gi]) else 0.0
        records.append(dict(
            qid=q.qid, text=q.text, mm=q.is_multimodal,
            gold=sorted(gold), gold_rels=gold_rels, gold_has_img=gold_has_img,
            n_gold=len(gold), rank_morse=rank_g, rank_fusion=rank_f,
            hit_morse=hit_g, hit_fusion=hit_f,
            ndcg_morse=100 * ndcg_at_k(rg, gold, K), ndcg_fusion=100 * ndcg_at_k(rf, gold, K),
            top_morse=rg[:K], top_fusion=rf[:K], alpha_gold=alpha,
        ))

    # 1) outcome matrix ------------------------------------------------------
    both = sum(r["hit_morse"] and r["hit_fusion"] for r in records)
    wins = [r for r in records if r["hit_morse"] and not r["hit_fusion"]]
    losses = [r for r in records if r["hit_fusion"] and not r["hit_morse"]]
    neither = sum(not r["hit_morse"] and not r["hit_fusion"] for r in records)
    n = len(records)

    # 2) breakdowns ----------------------------------------------------------
    def rate(sub):
        return (100 * sum(r["hit_morse"] for r in sub) / len(sub),
                100 * sum(r["hit_fusion"] for r in sub) / len(sub), len(sub)) if sub else (0, 0, 0)

    by_mod = {m: rate([r for r in records if r["mm"] == m]) for m in [True, False]}
    by_img = {g: rate([r for r in records if r["gold_has_img"] == g]) for g in [True, False]}
    by_ngold = {"1 gold": rate([r for r in records if r["n_gold"] == 1]),
                "2+ gold": rate([r for r in records if r["n_gold"] >= 2])}
    rel_counter = Counter(rel for r in records for rel in set(r["gold_rels"]))
    by_rel = {}
    for rel, cnt in rel_counter.most_common(12):
        sub = [r for r in records if rel in r["gold_rels"]]
        by_rel[rel] = rate(sub)

    # 3) rank movement -------------------------------------------------------
    moves = [(r["rank_fusion"] or args.depth + 1) - (r["rank_morse"] or args.depth + 1)
             for r in records]
    improved = sum(m > 0 for m in moves)
    worsened = sum(m < 0 for m in moves)
    unchanged = sum(m == 0 for m in moves)

    # 4) markdown report -----------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    L = []
    L.append(f"# Error analysis: MORSE vs Fusion — {os.path.basename(args.data)}, "
             f"{args.setting}, {args.split}, hit@{K}\n")
    L.append(f"Config: beta={args.beta}, gamma={args.gamma}, lambda={args.lam}; "
             f"{n} evaluated queries.\n")
    L.append("## 1. Outcome matrix (first gold triplet in top-%d)\n" % K)
    L.append(f"| Outcome | Count | % |\n|---|---|---|")
    for name, v in [("Both correct", both), ("MORSE only (wins)", len(wins)),
                    ("Fusion only (losses)", len(losses)), ("Both wrong", neither)]:
        L.append(f"| {name} | {v} | {100*v/n:.1f}% |")
    L.append(f"\nNet: **{len(wins) - len(losses):+d}** queries in MORSE's favor.\n")

    def table(title, d):
        L.append(f"## {title}\n")
        L.append("| Group | MORSE hit@%d | Fusion hit@%d | n |\n|---|---|---|---|" % (K, K))
        for k, (g, f, c) in d.items():
            name = {True: "multimodal", False: "text-only"}.get(k, str(k))
            L.append(f"| {name} | {g:.1f}% | {f:.1f}% | {c} |")
        L.append("")

    table("2a. By query modality", by_mod)
    table("2b. By gold-triplet modality (gold has image?)", by_img)
    table("2c. By number of gold triplets", by_ngold)
    table("2d. By gold relation (12 most frequent)", by_rel)

    L.append("## 3. Rank movement of first gold (Fusion rank − MORSE rank)\n")
    L.append(f"- Gold ranked **higher** under MORSE: {improved} ({100*improved/n:.1f}%)")
    L.append(f"- Unchanged: {unchanged} ({100*unchanged/n:.1f}%)")
    L.append(f"- Gold ranked lower under MORSE: {worsened} ({100*worsened/n:.1f}%)")
    L.append(f"- Mean movement: {sum(moves)/n:+.2f} positions\n")

    def dump_examples(title, subset, key):
        L.append(f"## {title}\n")
        subset = sorted(subset, key=key)[: args.n_examples]
        for i, r in enumerate(subset, 1):
            L.append(f"### Example {i} (qid={r['qid']}, {'multimodal' if r['mm'] else 'text'} query)")
            L.append(f"**Query:** {r['text']}")
            L.append(f"**Gold:** " + "; ".join(fmt_triplet(mkg, g) for g in r["gold"][:3]))
            L.append(f"**Gold rank:** MORSE={r['rank_morse']}, Fusion={r['rank_fusion']}; "
                     f"RCA gate on gold alpha={r['alpha_gold']:.2f}")
            L.append("**MORSE top-%d:**" % K)
            for j, idx in enumerate(r["top_morse"], 1):
                mark = " <-- GOLD" if idx in r["gold"] else ""
                L.append(f"  {j}. {fmt_triplet(mkg, idx)}{mark}")
            L.append("**Fusion top-%d:**" % K)
            for j, idx in enumerate(r["top_fusion"], 1):
                mark = " <-- GOLD" if idx in r["gold"] else ""
                L.append(f"  {j}. {fmt_triplet(mkg, idx)}{mark}")
            L.append("")

    dump_examples("4a. Qualitative — MORSE wins (Fusion missed, MORSE found)",
                  wins, key=lambda r: (r["rank_morse"] or 999))
    dump_examples("4b. Qualitative — MORSE losses (Fusion found, MORSE missed)",
                  losses, key=lambda r: (r["rank_fusion"] or 999))
    dump_examples("4c. Qualitative — both wrong (hardest queries)",
                  [r for r in records if not r["hit_morse"] and not r["hit_fusion"]],
                  key=lambda r: -(r["n_gold"]))

    report = "\n".join(L)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(report)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, default=int)
    print(f"\nWrote {args.out}.md and {args.out}.json")
    print(f"Summary: both={both}, wins={len(wins)}, losses={len(losses)}, "
          f"neither={neither} (net {len(wins)-len(losses):+d})")


if __name__ == "__main__":
    main()
