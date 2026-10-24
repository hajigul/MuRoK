"""Attribute each wrong answer to retrieval failure (gold not in top-K) or
generation failure (gold retrieved, answer still wrong), and measure
parametric answering (correct despite no gold retrieved).

  python scripts/gen_error_breakdown.py --data data/bench_g --split test \
      --gen outputs/gen_g_s4_gram_tuned.json --name "MORSE (Bench-G S4)"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morse.data import load_queries
from morse.generator import contains_at_1, exact_match

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--gen", required=True)
ap.add_argument("--name", default="")
ap.add_argument("--correct", choices=["em", "contains"], default="contains",
                help="correctness criterion for the breakdown")
args = ap.parse_args()

gold_map = {q.qid: set(q.gold_triplets) for q in load_queries(args.data, args.split)}
records = json.load(open(args.gen, encoding="utf-8"))["records"]

is_ok = (lambda r: exact_match(r["pred"], r["answers"])) if args.correct == "em" \
    else (lambda r: exact_match(r["pred"], r["answers"]) or contains_at_1(r["pred"], r["answers"]))

n = ok_hit = ok_miss = bad_ret = bad_gen = skipped = 0
for r in records:
    gold = gold_map.get(r["qid"])
    if gold is None:
        skipped += 1
        continue
    hit = bool(gold & set(r["retrieved"]))
    ok = is_ok(r)
    n += 1
    if ok and hit: ok_hit += 1
    elif ok and not hit: ok_miss += 1
    elif not ok and hit: bad_gen += 1
    else: bad_ret += 1

label = args.name or os.path.basename(args.gen)
print(f"\n{label}  (n={n}, criterion={args.correct})")
print(f"  correct, gold retrieved (RAG success)        : {ok_hit:5d}  ({100*ok_hit/n:5.1f}%)")
print(f"  correct, gold NOT retrieved (parametric)     : {ok_miss:5d}  ({100*ok_miss/n:5.1f}%)")
print(f"  wrong,   gold retrieved (GENERATION failure) : {bad_gen:5d}  ({100*bad_gen/n:5.1f}%)")
print(f"  wrong,   gold NOT retrieved (RETRIEVAL failure): {bad_ret:5d}  ({100*bad_ret/n:5.1f}%)")
if skipped:
    print(f"  [skipped {skipped} records without gold labels]")
