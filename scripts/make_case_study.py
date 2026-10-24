"""Build a case-study table (markdown + LaTeX) from a saved generation run.

Workflow:
  1. Run generation and save records:
     python scripts/run_generation.py --data data/bench_g --split test --setting S4 \
         --method morse --limit 200 --save outputs/gen_g_s4.json
  2. Build the case-study table:
     python scripts/make_case_study.py --gen outputs/gen_g_s4.json \
         --data data/bench_g --out outputs/case_study_g_s4

Selects a mix of examples: correct answers grounded in a retrieved gold
triplet (the showcase rows), plus optionally some failure rows for the
error-analysis discussion (--include-failures).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morse.data import MKG
from morse.generator import exact_match, contains_at_1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="JSON saved by run_generation.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=8, help="number of showcase rows")
    ap.add_argument("--include-failures", type=int, default=3)
    ap.add_argument("--out", default="outputs/case_study")
    args = ap.parse_args()

    with open(args.gen, encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"]
    mkg = MKG.load(args.data)

    def verb(tid):
        c = mkg.triplets[mkg.tid2idx[tid]]
        tag = " [image]" if c.image else ""
        return f"({c.h_name}, {c.r_name}, {c.t_name}){tag}"

    successes, failures = [], []
    for r in records:
        ok = exact_match(r["pred"], r["answers"]) or contains_at_1(r["pred"], r["answers"])
        (successes if ok else failures).append(r)

    rows = successes[: args.n] + failures[: args.include_failures]
    print(f"{len(successes)} correct / {len(failures)} incorrect in the run; "
          f"showing {min(args.n, len(successes))} successes + "
          f"{min(args.include_failures, len(failures))} failures")

    # markdown ---------------------------------------------------------------
    md = ["# MORSE case study: question -> retrieved knowledge -> answer\n",
          f"Generation metrics of this run: {payload['metrics']}\n",
          "| # | Question | Top retrieved triplet (MORSE) | Model answer | Gold answer | Correct |",
          "|---|---|---|---|---|---|"]
    tex = ["\\begin{table*}[t]\\centering\\small",
           "\\caption{Case study: questions answered by the frozen generator "
           "conditioned on MORSE-retrieved triplets.}",
           "\\begin{tabular}{p{0.28\\textwidth}p{0.3\\textwidth}p{0.15\\textwidth}p{0.13\\textwidth}c}",
           "\\toprule Question & Top retrieved triplet & Model answer & Gold & Correct\\\\ \\midrule"]
    for i, r in enumerate(rows, 1):
        top = verb(r["retrieved"][0]) if r["retrieved"] else "-"
        ok = "yes" if exact_match(r["pred"], r["answers"]) or contains_at_1(r["pred"], r["answers"]) else "no"
        gold = "; ".join(r["answers"][:2])
        md.append(f"| {i} | {r['question']} | {top} | {r['pred']} | {gold} | {ok} |")
        esc = lambda s: str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        tex.append(f"{esc(r['question'])} & {esc(top)} & {esc(r['pred'])} & {esc(gold)} & {ok}\\\\")
    tex += ["\\bottomrule\\end{tabular}\\end{table*}"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(args.out + ".tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"Wrote {args.out}.md and {args.out}.tex")


if __name__ == "__main__":
    main()
