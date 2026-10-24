"""Generate the full LaTeX results table directly from run_retrieval.py's
saved JSON, so every number in the paper is copied programmatically from
the measured results (no manual transcription).

  python scripts/make_latex_table.py --json outputs/fb15k237img_final_tuned.json \
      --out outputs/fb15k237img_table.tex

Multiple --json files can be given (merged; later files win on key clashes),
e.g. to combine a main run with a separately saved random-baseline run.
"""
from __future__ import annotations

import argparse
import json

SETTING_LABELS = {
    "S1": ("All", "All"),
    "S2": ("Text-only", "Text-only"),
    "S3": ("Text-only", "All"),
    "S4": ("Multimodal", "Multimodal"),
    "S5": ("Multimodal", "All"),
}
METHOD_ORDER = ["random", "text-only", "captioning", "fusion", "reranking", "morse"]
METHOD_LABELS = {
    "random": "Random",
    "text-only": "Text-only",
    "captioning": "Captioning",
    "fusion": "Fusion",
    "reranking": "Reranking",
    "morse": "MORSE (Ours)",
}
KS = [5, 10, 20, 50, 100]
COLS = [f"NDCG@{k}" for k in KS] + [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS]

CAPTION = (
    "Retrieval on FB15k-237-IMG (test split) with queries constructed via the "
    "MKG-RAG-Bench masking protocol (tail-masked triplets; gold = all corpus "
    "triplets sharing the query's head and relation; multimodal queries attach "
    "the head-entity image). All six methods are implemented in our pipeline "
    "over the same frozen CLIP ViT-B/32 encoders; the Captioning baseline "
    "replaces images with BLIP-generated captions and retrieves by text, and "
    "the Reranking baseline performs visual-first stage-one retrieval followed "
    "by cross-modal reranking of the top-200. MORSE's gate temperature "
    "($\\beta{=}12$, $\\gamma{=}0$, $\\lambda{=}1.0$) is selected on the "
    "validation split only. The text-only query settings contain only "
    "$n{=}23$ queries on this dataset (98.7\\% of entities carry an image) "
    "and should be interpreted accordingly. Best per column in \\textbf{bold}; "
    "second best \\underline{underlined}."
)


def fmt(v, best, second):
    s = f"{v:.2f}"
    if abs(v - best) < 5e-3:
        return f"\\textbf{{{s}}}"
    if abs(v - second) < 5e-3:
        return f"\\underline{{{s}}}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="tab:fb15k237img")
    args = ap.parse_args()

    R = {}
    for path in args.json:
        with open(path, encoding="utf-8") as f:
            R.update(json.load(f))

    L = []
    L.append("\\begin{table*}[t]\n\\centering")
    L.append(f"\\caption{{{CAPTION}}}")
    L.append(f"\\label{{{args.label}}}")
    L.append("\\resizebox{\\textwidth}{!}{%")
    L.append("\\begin{tabular}{ll l " + "c" * 15 + "}")
    L.append("\\toprule")
    L.append("\\multirow{2}{*}{Query} & \\multirow{2}{*}{Triplet} & \\multirow{2}{*}{Method}")
    L.append(" & \\multicolumn{5}{c}{NDCG@$K$ $\\uparrow$}"
             " & \\multicolumn{5}{c}{Precision@$K$ $\\uparrow$}"
             " & \\multicolumn{5}{c}{Recall@$K$ $\\uparrow$} \\\\")
    L.append("\\cmidrule(lr){4-8}\\cmidrule(lr){9-13}\\cmidrule(lr){14-18}")
    L.append(" & & & " + " & ".join(str(k) for k in KS * 3) + " \\\\")

    for setting in ["S1", "S2", "S3", "S4", "S5"]:
        methods = [m for m in METHOD_ORDER if f"{setting}/{m}" in R]
        if not methods:
            continue
        L.append("\\midrule")
        q_lab, t_lab = SETTING_LABELS[setting]
        # per-column best/second across the methods in this block
        best, second = {}, {}
        for col in COLS:
            vals = sorted((R[f"{setting}/{m}"][col] for m in methods), reverse=True)
            best[col] = vals[0]
            second[col] = vals[1] if len(vals) > 1 else float("-inf")
        n = len(methods)
        for i, m in enumerate(methods):
            row = R[f"{setting}/{m}"]
            cells = " & ".join(fmt(row[col], best[col], second[col]) for col in COLS)
            if i == 0:
                prefix = (f"\\multirow{{{n}}}{{*}}{{{q_lab}}} & "
                          f"\\multirow{{{n}}}{{*}}{{{t_lab}}} & {METHOD_LABELS[m]}")
            else:
                prefix = f" & & {METHOD_LABELS[m]}"
            L.append(f"{prefix} & {cells} \\\\")

    L.append("\\bottomrule\n\\end{tabular}}\n\\end{table*}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")
    # quick console summary of NDCG@5 for verification
    for setting in ["S1", "S2", "S3", "S4", "S5"]:
        row = {m: R[f"{setting}/{m}"]["NDCG@5"]
               for m in METHOD_ORDER if f"{setting}/{m}" in R}
        if row:
            print(setting, {METHOD_LABELS[m]: round(v, 2) for m, v in row.items()})


if __name__ == "__main__":
    main()
