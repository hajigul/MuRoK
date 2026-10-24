"""Generate publication-ready error-analysis plots from the JSON written by
scripts/error_analysis.py.

  python scripts/plot_error_analysis.py --json outputs/error_analysis_g_s4.json \
      --out outputs/plots_g_s4

Produces four PNG/PDF figures:
  1_outcome_matrix      MORSE vs Fusion win/loss/tie bar chart
  2_hit_by_relation     hit@K by gold relation type (MORSE vs Fusion, top 10)
  3_rank_movement       histogram of gold-rank movement (Fusion rank - MORSE rank)
  4_hit_by_modality     hit@K by query modality and by gold-triplet modality
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}_{name}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="outputs/plots")
    ap.add_argument("--K", type=int, default=5)
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        R = json.load(f)
    n = len(R)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- 1. outcome matrix -------------------------------------------------
    both = sum(r["hit_morse"] and r["hit_fusion"] for r in R)
    wins = sum(r["hit_morse"] and not r["hit_fusion"] for r in R)
    losses = sum(r["hit_fusion"] and not r["hit_morse"] for r in R)
    neither = n - both - wins - losses
    fig, ax = plt.subplots(figsize=(5, 3.2))
    cats = ["Both\ncorrect", "MORSE only\n(wins)", "Fusion only\n(losses)", "Both\nwrong"]
    vals = [both, wins, losses, neither]
    colors = ["#4C9F70", "#2E7DD1", "#D1662E", "#999999"]
    bars = ax.bar(cats, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100*v/n:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"# queries (hit@{args.K})")
    ax.set_title(f"MORSE vs Fusion outcome matrix (n={n})")
    ax.set_ylim(0, max(vals) * 1.25)
    save(fig, args.out, "1_outcome_matrix")

    # ---- 2. hit@K by relation ---------------------------------------------
    rel_counter = Counter(rel for r in R for rel in set(r["gold_rels"]))
    rels = [rel for rel, _ in rel_counter.most_common(10)]
    g_rates, f_rates, ns = [], [], []
    for rel in rels:
        sub = [r for r in R if rel in r["gold_rels"]]
        g_rates.append(100 * sum(r["hit_morse"] for r in sub) / len(sub))
        f_rates.append(100 * sum(r["hit_fusion"] for r in sub) / len(sub))
        ns.append(len(sub))
    x = np.arange(len(rels))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - 0.2, g_rates, 0.4, label="MORSE", color="#2E7DD1")
    ax.bar(x + 0.2, f_rates, 0.4, label="Fusion", color="#D1662E")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[:18]}\n(n={c})" for r, c in zip(rels, ns)],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(f"hit@{args.K} (%)")
    ax.set_title("Retrieval success by gold relation type")
    ax.legend()
    save(fig, args.out, "2_hit_by_relation")

    # ---- 3. rank movement ---------------------------------------------------
    depth_pen = 101
    moves = [(r["rank_fusion"] or depth_pen) - (r["rank_morse"] or depth_pen) for r in R]
    moves_nz = [m for m in moves if m != 0]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if moves_nz:
        lim = max(1, min(30, int(np.percentile(np.abs(moves_nz), 95))))
        clipped = np.clip(moves_nz, -lim, lim)
        ax.hist(clipped, bins=np.arange(-lim - 0.5, lim + 1.5, 1),
                color="#2E7DD1", edgecolor="white")
    ax.axvline(0, color="k", lw=0.8)
    up = sum(m > 0 for m in moves)
    dn = sum(m < 0 for m in moves)
    ax.set_xlabel("Gold-rank movement (Fusion rank − MORSE rank); >0 = MORSE ranks gold higher")
    ax.set_ylabel("# queries")
    ax.set_title(f"Rank movement of first gold: improved {up}, worsened {dn}, "
                 f"unchanged {n-up-dn}")
    save(fig, args.out, "3_rank_movement")

    # ---- 4. modality breakdown ---------------------------------------------
    groups = []
    for label, pred in [("MM query", lambda r: r["mm"]),
                        ("Text query", lambda r: not r["mm"]),
                        ("Gold has image", lambda r: r["gold_has_img"]),
                        ("Gold text-only", lambda r: not r["gold_has_img"])]:
        sub = [r for r in R if pred(r)]
        if sub:
            groups.append((label,
                           100 * sum(r["hit_morse"] for r in sub) / len(sub),
                           100 * sum(r["hit_fusion"] for r in sub) / len(sub),
                           len(sub)))
    if groups:
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(x - 0.2, [g[1] for g in groups], 0.4, label="MORSE", color="#2E7DD1")
        ax.bar(x + 0.2, [g[2] for g in groups], 0.4, label="Fusion", color="#D1662E")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g[0]}\n(n={g[3]})" for g in groups], fontsize=8)
        ax.set_ylabel(f"hit@{args.K} (%)")
        ax.set_title("Retrieval success by query / gold modality")
        ax.legend()
        save(fig, args.out, "4_hit_by_modality")


if __name__ == "__main__":
    main()
