"""Generate the end-to-end generation LaTeX table directly from the JSON
files saved by run_generation.py, so every cell is copied programmatically.

Each --entry has the form  DATASETKEY|SETTING|METHOD|path.json
  DATASETKEY : short key, must match one of the --datasets definitions
  SETTING    : S1..S5 (mapped to Query x Triplet labels)
  METHOD     : Fusion | MORSE | All  ("All" = single shared row for
               text-only query settings where retrieval coincides)

Example:
  python scripts/make_gen_table.py --out outputs/generation_table.tex ^
    --datasets "G|MKG-RAG-Bench-G" "FB|FB15k-237-IMG" ^
    --entry "G|S1|Fusion|outputs/gen_g_s1_fusion.json" ^
    --entry "G|S1|MORSE|outputs/gen_g_s1_gram_tuned.json" ^
    ... (one per run)
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
METRICS = ["EM", "F1", "Contains@1", "BLEU-1"]
HEADERS = ["EM", "F1", "Cont.@1", "BLEU-1"]
METHOD_ORDER = ["All", "Fusion", "MORSE"]
METHOD_LABELS = {"All": "All methods$^{*}$", "Fusion": "Fusion",
                 "MORSE": "MORSE (Ours)"}

CAPTION = (
    "End-to-end generation (test splits) with $K{=}5$ retrieved triplets and "
    "a frozen \\texttt{Qwen2.5-VL-7B-Instruct} generator (identical "
    "generator, prompt, and decoding for all rows). MORSE uses each dataset's "
    "validation-selected configuration. In text-only query settings all "
    "retrievers produce coinciding retrieval and are reported as one row "
    "($^{*}$). Best per block in \\textbf{bold}."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help='e.g. "G|MKG-RAG-Bench-G" "FB|FB15k-237-IMG"')
    ap.add_argument("--entry", action="append", required=True,
                    help="DATASETKEY|SETTING|METHOD|path.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="tab:generation")
    args = ap.parse_args()

    ds_keys, ds_names = [], {}
    for d in args.datasets:
        k, name = d.split("|", 1)
        ds_keys.append(k)
        ds_names[k] = name

    cell = {}   # (dskey, setting, method) -> metrics dict
    for e in args.entry:
        k, setting, method, path = e.split("|", 3)
        with open(path, encoding="utf-8") as f:
            cell[(k, setting, method)] = json.load(f)["metrics"]

    def block_methods(setting):
        ms = []
        for m in METHOD_ORDER:
            if any((k, setting, m) in cell for k in ds_keys):
                ms.append(m)
        return ms

    ncols = 3 + 4 * len(ds_keys)
    L = ["\\begin{table*}[t]\n\\centering",
         f"\\caption{{{CAPTION}}}",
         f"\\label{{{args.label}}}",
         "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{ll l " + " ".join(["rrrr"] * len(ds_keys)) + "}",
         "\\toprule",
         "\\multirow{2}{*}{Query} & \\multirow{2}{*}{Triplet} & "
         "\\multirow{2}{*}{Method}"]
    for i, k in enumerate(ds_keys):
        L[-1] += f" & \\multicolumn{{4}}{{c}}{{{ds_names[k]}}}"
    L[-1] += " \\\\"
    cm = []
    for i in range(len(ds_keys)):
        a = 4 + 4 * i
        cm.append(f"\\cmidrule(lr){{{a}-{a+3}}}")
    L.append("".join(cm))
    L.append(" & & & " + " & ".join(
        " & ".join(HEADERS) for _ in ds_keys) + " \\\\")

    for setting in ["S1", "S2", "S3", "S4", "S5"]:
        ms = block_methods(setting)
        if not ms:
            continue
        L.append("\\midrule")
        q_lab, t_lab = SETTING_LABELS[setting]
        # best per dataset+metric within this block
        best = {}
        for k in ds_keys:
            for met in METRICS:
                vals = [cell[(k, setting, m)][met] for m in ms
                        if (k, setting, m) in cell]
                best[(k, met)] = max(vals) if vals else None
        n = len(ms)
        for i, m in enumerate(ms):
            cells = []
            for k in ds_keys:
                if (k, setting, m) in cell:
                    for met in METRICS:
                        v = cell[(k, setting, m)][met]
                        s = f"{v:.2f}"
                        if len(ms) > 1 and best[(k, met)] is not None and \
                                abs(v - best[(k, met)]) < 5e-3:
                            s = f"\\textbf{{{s}}}"
                        cells.append(s)
                else:
                    cells.extend(["--"] * 4)
            prefix = (f"\\multirow{{{n}}}{{*}}{{{q_lab}}} & "
                      f"\\multirow{{{n}}}{{*}}{{{t_lab}}} & "
                      f"{METHOD_LABELS[m]}") if i == 0 else \
                     f" & & {METHOD_LABELS[m]}"
            L.append(prefix + " & " + " & ".join(cells) + " \\\\")

    L.append("\\bottomrule\n\\end{tabular}}\n\\end{table*}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}\n")
    # console summary (EM) for verification
    for setting in ["S1", "S2", "S3", "S4", "S5"]:
        for m in METHOD_ORDER:
            row = {k: round(cell[(k, setting, m)]["EM"], 2)
                   for k in ds_keys if (k, setting, m) in cell}
            if row:
                print(f"{setting:3s} {m:6s} EM: {row}")


if __name__ == "__main__":
    main()
