"""Download MKG-RAG-Bench from the official repo and convert it to this
codebase's format — fully automatic for all text data and gold labels.

  python scripts/download_benchmark.py --out data                 # both subsets
  python scripts/download_benchmark.py --out data --subset g      # Bench-G only
  python scripts/download_benchmark.py --out data --subset m --mimic-root /path/to/physionet.org

What gets downloaded automatically:
  * Bench-G and Bench-M: all queries, all triplets (text + multimodal corpora),
    exact gold retrieval labels (qrels), for train/val/test — straight from
    https://github.com/XiaochenWang-PSU/MKG-RAG-Bench

Images:
  * Bench-G images come from the MARS dataset (MKG_Analogy, ICLR'23 release,
    Google Drive). Pass --download-mars-images to fetch them with gdown, or
    point --mars-images at an existing MARS/images directory.
  * Bench-M images are MIMIC-CXR-JPG chest X-rays hosted on PhysioNet, which
    REQUIRES a credentialed PhysioNet account and license — they cannot be
    fetched anonymously. Point --mimic-root at your local copy (the directory
    containing physionet.org/files/mimic-cxr-jpg/...).
  * Without images you can still run every text-side experiment (S2/S3) and
    all retrieval structure; S4/S5 need the images.

Output format (per subset directory, e.g. data/bench_g/):
  entities.json / relations.json / triplets.json / queries_{split}.json
Triplet ids are namespaced: 'txt{doc_id}' for the text corpus and
'mm{doc_id}' for the multimodal corpus, matching the qrels doc-id spaces.
Gold answers for generation are derived from the masked field of each gold
triplet (tail_text for masked_type=tail, head_text for masked_type=head).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/XiaochenWang-PSU/MKG-RAG-Bench.git"
MARS_GDRIVE_ID = "1AqnyrA05vKngfEbhw1mxY5qEoaqiKsC1"  # from zjunlp/MKG_Analogy README

SUBSETS = {
    "g": {"src": "MKG-RAG-BENCH-G", "out": "bench_g", "splits": {"train": "train", "val": "valid", "test": "test"}},
    "m": {"src": "MKG-RAG-BENCH-M", "out": "bench_m", "splits": {"train": "train", "val": "val", "test": "test"}},
}


def sh(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def clone_repo(cache_dir: str) -> str:
    repo = os.path.join(cache_dir, "MKG-RAG-Bench")
    if os.path.exists(os.path.join(repo, "MKG-RAG-BENCH-G")):
        print(f"Using existing clone: {repo}")
        return repo
    os.makedirs(cache_dir, exist_ok=True)
    sh(["git", "clone", "--depth", "1", REPO_URL, repo])
    return repo


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_qrels(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                qid, did = parts[0], parts[1]
                gold.setdefault(str(qid), []).append(did)
    return gold


def resolve_image(raw_path, subset, mars_images, mimic_root):
    """Map a corpus image_path to an absolute path on this machine (or None)."""
    if not raw_path:
        return None
    if subset == "g":
        if mars_images:
            p = os.path.join(mars_images, raw_path)
            return p if os.path.exists(p) else None
        return None
    # Bench-M: paths look like /physionet.org/files/mimic-cxr-jpg/2.0.0/files/...
    if mimic_root:
        p = os.path.join(mimic_root, raw_path.lstrip("/"))
        return p if os.path.exists(p) else None
    return None


def convert_subset(repo, key, out_root, mars_images, mimic_root):
    spec = SUBSETS[key]
    src = os.path.join(repo, spec["src"])
    out = os.path.join(out_root, spec["out"])
    os.makedirs(out, exist_ok=True)

    # Corpora are identical across splits (verified: same file sizes/content);
    # read them once from the test split.
    ref_split = spec["splits"]["test"]
    txt_corpus = read_jsonl(os.path.join(src, ref_split, "text_corpus.jsonl"))
    mm_corpus = read_jsonl(os.path.join(src, ref_split, "mm_corpus.jsonl"))

    entities, relations, triplets = {}, {}, []
    head2img = {}       # head_id -> resolved query-image path (for mm queries)
    tid_info = {}       # tid -> (head_text, tail_text) for answer derivation
    img_found = img_total = 0

    def reg_entity(eid, name):
        if eid not in entities:
            entities[eid] = {"name": name, "image": None}

    for prefix, corpus, is_mm in [("txt", txt_corpus, False), ("mm", mm_corpus, True)]:
        for row in corpus:
            reg_entity(row["head_id"], row["head_text"])
            reg_entity(row["tail_id"], row["tail_text"])
            relations.setdefault(row["rel_id"], row["rel_text"])
            tid = f"{prefix}{row['doc_id']}"
            item = {"id": tid, "h": row["head_id"], "r": row["rel_id"], "t": row["tail_id"]}
            if is_mm:
                item["mm"] = True
                img_total += 1
                abs_img = resolve_image(row.get("image_path"), key, mars_images, mimic_root)
                if abs_img:
                    item["image"] = os.path.relpath(abs_img, out)
                    head2img.setdefault(row["head_id"], item["image"])
                    img_found += 1
                else:
                    # keep the raw path so a later image download can be re-linked
                    item["image_raw"] = row.get("image_path")
            triplets.append(item)
            tid_info[tid] = (row["head_text"], row["tail_text"])

    # NOTE on multimodality without local images: if images are missing we
    # leave item["image"] unset, so those triplets load as text-only. Re-run
    # this script after fetching images to re-link them.

    for split, src_split in spec["splits"].items():
        out_q = []
        for kind, prefix in [("text", "txt"), ("mm", "mm")]:
            qpath = os.path.join(src, src_split, f"{kind}_queries.jsonl")
            rpath = os.path.join(src, src_split, f"{kind}_qrels.tsv")
            if not os.path.exists(qpath):
                continue
            gold_map = read_qrels(rpath) if os.path.exists(rpath) else {}
            for row in read_jsonl(qpath):
                qid = str(row["qid"])
                gold = [f"{prefix}{d}" for d in gold_map.get(qid, [])]
                masked = row.get("masked_type", "tail")
                answers = []
                for g in gold:
                    if g in tid_info:
                        h_txt, t_txt = tid_info[g]
                        answers.append(t_txt if masked == "tail" else h_txt)
                answers = sorted(set(answers))
                qimg = None
                if row.get("is_multimodal"):
                    qimg = head2img.get(row.get("head_id"))
                text = row["query"].replace("[IMAGE]", "").strip()
                out_q.append({
                    "mm": bool(row.get("is_multimodal")),
                    "id": f"{kind}_{qid}",
                    "text": text,
                    "image": qimg,
                    "gold_triplets": gold,
                    "question": text,
                    "answers": answers,
                })
        with open(os.path.join(out, f"queries_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(out_q, f, ensure_ascii=False)
        n_mm = sum(1 for q in out_q if q["image"])
        print(f"  {split}: {len(out_q)} queries ({n_mm} with resolved images)")

    json.dump(entities, open(os.path.join(out, "entities.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(relations, open(os.path.join(out, "relations.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(triplets, open(os.path.join(out, "triplets.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  triplets: {len(triplets)} ({len(mm_corpus)} multimodal, "
          f"{img_found}/{img_total} images resolved locally)")
    print(f"  -> {out}")
    if img_found == 0 and img_total > 0:
        if key == "g":
            print("  [note] No Bench-G images found. Fetch MARS images with "
                  "--download-mars-images (or --mars-images /path/to/MARS/images) "
                  "and re-run to enable S4/S5.")
        else:
            print("  [note] No Bench-M images found. MIMIC-CXR-JPG requires a "
                  "credentialed PhysioNet account; download it there and re-run "
                  "with --mimic-root /path/containing/physionet.org")


def download_mars_images(dest: str) -> str:
    try:
        import gdown  # noqa
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "gdown"])
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "MARS_images.zip")
    if not os.path.exists(zip_path):
        sh(["gdown", MARS_GDRIVE_ID, "-O", zip_path])
    sh([sys.executable, "-m", "zipfile", "-e", zip_path, dest])
    # locate the images directory inside the extracted archive
    for cand in ["images", "MARS/images", "MarT/dataset/MARS/images"]:
        p = os.path.join(dest, cand)
        if os.path.isdir(p):
            return p
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--subset", choices=["g", "m", "both"], default="both")
    ap.add_argument("--cache", default="data/_raw", help="where to clone the official repo")
    ap.add_argument("--mars-images", default=None,
                    help="existing MARS images dir (contains Q*/*.jpg folders)")
    ap.add_argument("--download-mars-images", action="store_true",
                    help="fetch MARS images from Google Drive via gdown (~several GB)")
    ap.add_argument("--mimic-root", default=None,
                    help="directory containing physionet.org/files/mimic-cxr-jpg/...")
    args = ap.parse_args()

    repo = clone_repo(args.cache)
    mars = args.mars_images
    if args.download_mars_images and not mars:
        mars = download_mars_images(os.path.join(args.cache, "mars"))
        print(f"MARS images at: {mars}")

    keys = ["g", "m"] if args.subset == "both" else [args.subset]
    for k in keys:
        print(f"\nConverting Bench-{k.upper()} ...")
        convert_subset(repo, k, args.out, mars, args.mimic_root)

    print("\nDone. Next:")
    print("  python scripts/run_retrieval.py --data data/bench_g --split test --settings S1 S2 S3 S4 S5")


if __name__ == "__main__":
    main()
