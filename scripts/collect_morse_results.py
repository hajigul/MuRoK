"""Collect saved MORSE retrieval results into a text summary + LaTeX table.

Reads the JSON files written by run_retrieval.py (--save), which are keyed
"S{n}/{method}" -> {NDCG@5.., P@5.., R@5.., QPS}. Writes:
  results/morse_summary.txt   human-readable, all settings x all K
  results/morse_table.tex     LaTeX (one block per setting, NDCG/P/R x K)

Usage:
  python scripts/collect_morse_results.py ^
    --in "MKG-RAG-Bench-G|results/retr_morse_bench_g.json" ^
         "FB15k-237-IMG|results/retr_morse_fb.json" ^
    --method morse ^
    --out-txt results/morse_summary.txt ^
    --out-tex results/morse_table.tex
"""
from __future__ import annotations
import argparse, json, os

KS = [5, 10, 20, 50, 100]
SETTINGS = ["S1", "S2", "S3", "S4", "S5"]
SETTING_LABEL = {
    "S1": ("All", "All"), "S2": ("Text-only", "Text-only"),
    "S3": ("Text-only", "All"), "S4": ("Multimodal", "Multimodal"),
    "S5": ("Multimodal", "All"),
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help='each "DatasetName|path.json"')
    ap.add_argument("--method", default="morse")
    ap.add_argument("--out-txt", default="results/morse_summary.txt")
    ap.add_argument("--out-tex", default="results/morse_table.tex")
    args = ap.parse_args()

    datasets = []
    for item in args.inputs:
        name, path = item.split("|", 1)
        datasets.append((name, load(path)))

    # ---------- text summary ----------
    lines = []
    for name, res in datasets:
        lines.append("=" * 70)
        lines.append(f"MORSE results — {name}")
        lines.append("=" * 70)
        for s in SETTINGS:
            key = f"{s}/{args.method}"
            if key not in res:
                continue
            m = res[key]
            q, t = SETTING_LABEL[s]
            lines.append(f"\n[{s}]  Query={q}  Triplet={t}"
                         f"   (QPS {m.get('QPS', float('nan')):.1f})")
            lines.append("      " + "".join(f"@{k:<8}" for k in KS))
            for metric in ("NDCG", "P", "R"):
                row = "".join(f"{m[f'{metric}@{k}']:<9.2f}" for k in KS)
                lines.append(f"  {metric:<4}{row}")
        lines.append("")
    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(txt)

    # ---------- LaTeX table ----------
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{MORSE retrieval results (test split). Router trained "
         "on the train split with best-validation checkpoint selection; "
         "frozen CLIP encoders and cached index shared across methods.}",
         "\\label{tab:morse_results}",
         "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{ll l ccccc ccccc ccccc}", "\\toprule",
         "\\multirow{2}{*}{Query} & \\multirow{2}{*}{Triplet} & "
         "\\multirow{2}{*}{Dataset}",
         " & \\multicolumn{5}{c}{NDCG@$K$ $\\uparrow$}"
         " & \\multicolumn{5}{c}{Precision@$K$ $\\uparrow$}"
         " & \\multicolumn{5}{c}{Recall@$K$ $\\uparrow$} \\\\",
         "\\cmidrule(lr){4-8}\\cmidrule(lr){9-13}\\cmidrule(lr){14-18}",
         " & & & 5 & 10 & 20 & 50 & 100 & 5 & 10 & 20 & 50 & 100 "
         "& 5 & 10 & 20 & 50 & 100 \\\\", "\\midrule"]
    for s in SETTINGS:
        q, t = SETTING_LABEL[s]
        rows = []
        for name, res in datasets:
            key = f"{s}/{args.method}"
            if key not in res:
                continue
            m = res[key]
            cells = "".join(f" & {m[f'NDCG@{k}']:.2f}" for k in KS) + \
                    "".join(f" & {m[f'P@{k}']:.2f}" for k in KS) + \
                    "".join(f" & {m[f'R@{k}']:.2f}" for k in KS)
            rows.append((name, cells))
        if not rows:
            continue
        span = len(rows)
        for i, (name, cells) in enumerate(rows):
            if i == 0:
                pre = (f"\\multirow{{{span}}}{{*}}{{{q}}} & "
                       f"\\multirow{{{span}}}{{*}}{{{t}}} & {name}")
            else:
                pre = f" & & {name}"
            L.append(pre + cells + " \\\\")
        L.append("\\midrule")
    if L[-1] == "\\midrule":
        L[-1] = "\\bottomrule"
    else:
        L.append("\\bottomrule")
    L += ["\\end{tabular}}", "\\end{table*}"]
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out_txt}\nwrote {args.out_tex}")
    print("\n--- text preview ---\n" + txt[:900])


if __name__ == "__main__":
    main()
