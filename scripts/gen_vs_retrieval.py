"""Join retrieval outcomes with generation outcomes per bucket
(both-hit / MORSE-only / Fusion-only / neither).

  python scripts/gen_vs_retrieval.py --ea outputs/error_analysis_g_s4.json \
      --gen-morse outputs/gen_g_s4_gram_tuned.json \
      --gen-fusion outputs/gen_g_s4_full_fusion.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morse.generator import exact_match, contains_at_1

ap = argparse.ArgumentParser()
ap.add_argument("--ea", required=True)
ap.add_argument("--gen-morse", required=True)
ap.add_argument("--gen-fusion", required=True)
args = ap.parse_args()

ea = {r["qid"]: r for r in json.load(open(args.ea, encoding="utf-8"))}
gg = {r["qid"]: r for r in json.load(open(args.gen_morse, encoding="utf-8"))["records"]}
gf = {r["qid"]: r for r in json.load(open(args.gen_fusion, encoding="utf-8"))["records"]}

BUCKETS = {(True, True): "both-hit", (True, False): "MORSE-only",
           (False, True): "Fusion-only", (False, False): "neither"}
stats, missing = {}, 0
for qid, r in ea.items():
    if qid not in gg or qid not in gf:
        missing += 1
        continue
    b = stats.setdefault(BUCKETS[(r["hit_morse"], r["hit_fusion"])],
                         {"n": 0, "em_g": 0, "em_f": 0, "c_g": 0, "c_f": 0})
    b["n"] += 1
    b["em_g"] += exact_match(gg[qid]["pred"], gg[qid]["answers"])
    b["em_f"] += exact_match(gf[qid]["pred"], gf[qid]["answers"])
    b["c_g"] += contains_at_1(gg[qid]["pred"], gg[qid]["answers"])
    b["c_f"] += contains_at_1(gf[qid]["pred"], gf[qid]["answers"])
if missing:
    print(f"[note] {missing} queries lacked generation records; skipped")
print(f"\n{'Bucket':<12} {'n':>6} | {'EM MORSE':>8} {'EM Fusion':>10} | "
      f"{'C@1 MORSE':>9} {'C@1 Fusion':>11}")
print("-" * 66)
for name in ["both-hit", "MORSE-only", "Fusion-only", "neither"]:
    if name not in stats:
        continue
    b = stats[name]; n = b["n"]
    print(f"{name:<12} {n:>6} | {100*b['em_g']/n:>8.1f} {100*b['em_f']/n:>10.1f} | "
          f"{100*b['c_g']/n:>9.1f} {100*b['c_f']/n:>11.1f}")
print(f"\nTotal joined: {sum(b['n'] for b in stats.values())}")
