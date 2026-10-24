"""Build an MKG-RAG-style evaluation from FB15k-237-IMG / WN18-IMG.

These datasets ship as raw KG triples + entity images, with NO queries or
retrieval labels. This script constructs them following the MKG-RAG-Bench
masking protocol:
  * corpus  = all triplets (train+valid+test), each verbalized "h | r | t";
              a triplet is multimodal if its head entity has an image
  * queries = sampled triplets with the TAIL masked:
              text  = "head_name | relation_name | [MASK]"
              image = head entity image (query is multimodal if present)
              gold  = ALL corpus triplets sharing (head, relation)   [1:N safe]
              answer= the masked tail entity name(s)
  * splits  : queries_val from valid triples, queries_test from test triples,
              queries_train from a sample of train triples

Expected inputs (MKGformer / RSME distribution layout):
  --triples-dir  folder containing train.txt / valid.txt / test.txt
                 (tab- or space-separated:  head_mid  relation  tail_mid)
  --entity2text  entity2text.txt or entity2textlong.txt (mid <TAB> name/desc)
  --images-dir   folder with entity images; entity folders/files may be named
                 like 'm.0xxxx' or '/m/0xxxx' -> sanitized automatically
Optional:
  --relation2text relation2text.txt (rel <TAB> name); otherwise names are
                  derived from the Freebase path (last segments, cleaned)

Usage:
  python scripts/build_mkg_eval.py \
      --triples-dir data/_raw/fb15k237/  \
      --entity2text data/_raw/fb15k237/entity2text.txt \
      --images-dir  data/_raw/fb15k237_images \
      --out data/fb15k237img --max-queries 2000
Then:
  python scripts/run_retrieval.py --data data/fb15k237img --split test \
      --settings S1 S2 S3 S4 S5 --methods morse fusion text-only random \
      --beta 5 --gamma 0 --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict


def load_triples(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) >= 3:
                out.append((parts[0], parts[1], parts[2]))
    return out


def load_map(path):
    m = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    # keep it short: first clause of long descriptions
                    m[parts[0]] = parts[1].split(",")[0].split(".")[0][:80]
    return m


def clean_relation(rel: str) -> str:
    """'/people/person/place_of_birth' -> 'person place of birth'."""
    segs = [s for s in re.split(r"[/.]", rel) if s]
    segs = segs[-2:] if len(segs) >= 2 else segs
    return " ".join(s.replace("_", " ") for s in segs) or rel


def sanitize(mid: str):
    """Candidate directory/file names an entity id might appear under."""
    yield mid
    yield mid.replace("/", ".").lstrip(".")
    yield mid.replace("/", "_").lstrip("_")
    yield mid.strip("/").replace("/", ".")
    yield mid.strip("/").replace("/", "_")


IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".JPEG", ".JPG", ".PNG")


def find_image(images_dir, mid):
    """Return path (relative to images_dir) of one image for this entity."""
    for cand in sanitize(mid):
        p = os.path.join(images_dir, cand)
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.endswith(IMG_EXT):
                    return os.path.join(cand, fn)
        for ext in IMG_EXT:
            if os.path.isfile(p + ext):
                return cand + ext
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples-dir", required=True)
    ap.add_argument("--entity2text", required=True)
    ap.add_argument("--relation2text", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-queries", type=int, default=2000,
                    help="max queries per split (val/test); train uses the same cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    splits = {}
    for name in ["train", "valid", "test"]:
        for fn in [f"{name}.txt", f"{name}.tsv"]:
            p = os.path.join(args.triples_dir, fn)
            if os.path.exists(p):
                splits[name] = load_triples(p)
                break
        if name not in splits:
            raise FileNotFoundError(f"missing {name}.txt/.tsv in {args.triples_dir}")

    ent_name = load_map(args.entity2text)
    rel_name_map = load_map(args.relation2text)

    all_triples = splits["train"] + splits["valid"] + splits["test"]
    print(f"triples: train={len(splits['train'])} valid={len(splits['valid'])} "
          f"test={len(splits['test'])} total={len(all_triples)}")

    # --- entities + images --------------------------------------------------
    entities, relations = {}, {}
    img_cache = {}
    n_img = 0
    os.makedirs(args.out, exist_ok=True)
    for h, r, t in all_triples:
        for e in (h, t):
            if e not in entities:
                img_rel = None
                if args.images_dir:
                    if e not in img_cache:
                        img_cache[e] = find_image(args.images_dir, e)
                    if img_cache[e]:
                        img_rel = os.path.relpath(
                            os.path.join(args.images_dir, img_cache[e]), args.out)
                        n_img += 1
                entities[e] = {"name": ent_name.get(e, e), "image": img_rel}
        relations.setdefault(r, rel_name_map.get(r, clean_relation(r)))
    print(f"entities: {len(entities)} ({n_img} with an image) | relations: {len(relations)}")

    # --- corpus triplets ----------------------------------------------------
    triplets = []
    hr2tids = defaultdict(list)  # (h, r) -> triplet ids, for 1:N gold sets
    for i, (h, r, t) in enumerate(all_triples):
        tid = f"t{i}"
        item = {"id": tid, "h": h, "r": r, "t": t}
        if entities[h]["image"]:
            item["mm"] = True
            item["image"] = entities[h]["image"]
        triplets.append(item)
        hr2tids[(h, r)].append(tid)

    n_mm = sum(1 for x in triplets if x.get("mm"))
    print(f"corpus: {len(triplets)} triplets ({n_mm} multimodal)")

    # --- queries (tail-masked) ----------------------------------------------
    def build_queries(triple_list, split, cap):
        idxs = list(range(len(triple_list)))
        rng.shuffle(idxs)
        out = []
        for j in idxs[:cap]:
            h, r, t = triple_list[j]
            gold = hr2tids[(h, r)]
            answers = sorted({entities[x_t]["name"]
                              for (x_h, x_r, x_t) in all_triples
                              if x_h == h and x_r == r})
            has_img = entities[h]["image"] is not None
            out.append({
                "id": f"{split}_{j}",
                "text": f"{entities[h]['name']} | {relations[r]} | [MASK]",
                "image": entities[h]["image"] if has_img else None,
                "mm": has_img,
                "gold_triplets": gold,
                "question": f"{entities[h]['name']} | {relations[r]} | [MASK]",
                "answers": answers,
            })
        return out

    for split_name, src in [("train", splits["train"]), ("val", splits["valid"]),
                            ("test", splits["test"])]:
        qs = build_queries(src, split_name, args.max_queries)
        with open(os.path.join(args.out, f"queries_{split_name}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(qs, f, ensure_ascii=False)
        n_q_mm = sum(1 for q in qs if q["mm"])
        print(f"queries_{split_name}: {len(qs)} ({n_q_mm} multimodal)")

    json.dump(entities, open(os.path.join(args.out, "entities.json"), "w",
                             encoding="utf-8"), ensure_ascii=False)
    json.dump(relations, open(os.path.join(args.out, "relations.json"), "w",
                              encoding="utf-8"), ensure_ascii=False)
    json.dump(triplets, open(os.path.join(args.out, "triplets.json"), "w",
                             encoding="utf-8"), ensure_ascii=False)
    print(f"-> {args.out}\nDescribe this in the paper as an evaluation "
          f"constructed from FB15k-237-IMG following the MKG-RAG-Bench "
          f"masking protocol (tail-masked queries; gold = all triplets "
          f"sharing the query's head and relation).")


if __name__ == "__main__":
    main()
