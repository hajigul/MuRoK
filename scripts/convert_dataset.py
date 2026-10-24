"""Convert your local copy of MKG-RAG-Bench data (MarKG-based Bench-G or
MedMKG-based Bench-M) into this repo's on-disk format.

Because dataset releases ship in slightly different layouts, this script
handles the two most common ones and is easy to adapt — the ONLY thing the
rest of the codebase needs is the format documented in morse/data.py:

  entities.json / relations.json / triplets.json / queries_{split}.json / images/

Supported input layouts:

(A) JSONL triples + JSON QA, e.g.
    --triples triples.jsonl        lines like {"head": ..., "relation": ..., "tail": ...}
    --entities entity2img.json     {entity_id: image_path}  (optional)
    --qa train.json val.json test.json

(B) TSV triples (head \t relation \t tail), same flags.

Example:
  python scripts/convert_dataset.py \
      --triples /path/markg/triples.tsv \
      --entity-images /path/markg/entity2image.json \
      --image-root /path/markg/images \
      --qa /path/bench_g/train.json /path/bench_g/val.json /path/bench_g/test.json \
      --out data/bench_g

QA records are mapped field-by-field; edit FIELD_MAP below if your release
uses different key names (this is deliberately explicit so you can see and
control exactly what maps where).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

# Adapt these to your QA release's key names if needed.
FIELD_MAP = {
    "id": ["id", "qid", "question_id"],
    "text": ["text", "query", "question"],
    "image": ["image", "query_image", "img"],
    "gold_triplets": ["gold_triplets", "gold", "positive_triplets", "labels"],
    "question": ["question", "text", "query"],
    "answers": ["answers", "answer", "gold_answers"],
}


def pick(d: dict, key: str, default=None):
    for k in FIELD_MAP[key]:
        if k in d and d[k] is not None:
            v = d[k]
            if key == "answers" and isinstance(v, str):
                return [v]
            return v
    return default


def load_triples(path: str):
    triples = []
    if path.endswith(".tsv") or path.endswith(".txt"):
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    triples.append({"h": parts[0], "r": parts[1], "t": parts[2]})
    else:  # json / jsonl
        with open(path) as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                raw = json.load(f)
            else:
                raw = [json.loads(l) for l in f if l.strip()]
        for x in raw:
            h = x.get("h") or x.get("head") or x.get("subject")
            r = x.get("r") or x.get("relation") or x.get("predicate")
            t = x.get("t") or x.get("tail") or x.get("object")
            triples.append({"h": h, "r": r, "t": t})
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True)
    ap.add_argument("--entity-images", default=None,
                    help="JSON mapping entity_id -> image filename")
    ap.add_argument("--entity-names", default=None,
                    help="JSON mapping entity_id -> display name (optional)")
    ap.add_argument("--relation-names", default=None,
                    help="JSON mapping relation_id -> display name (optional)")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--qa", nargs=3, metavar=("TRAIN", "VAL", "TEST"), required=True)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images into out/images (else store absolute paths)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    triples = load_triples(args.triples)

    ent_img = json.load(open(args.entity_images)) if args.entity_images else {}
    ent_name = json.load(open(args.entity_names)) if args.entity_names else {}
    rel_name = json.load(open(args.relation_names)) if args.relation_names else {}

    entities, relations, out_triples = {}, {}, []
    img_out = os.path.join(args.out, "images")
    if args.copy_images:
        os.makedirs(img_out, exist_ok=True)

    def register_entity(e):
        if e in entities:
            return
        img = ent_img.get(e)
        rel_path = None
        if img:
            src = os.path.join(args.image_root, img) if args.image_root else img
            if os.path.exists(src):
                if args.copy_images:
                    dst = os.path.join(img_out, os.path.basename(img))
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
                    rel_path = f"images/{os.path.basename(img)}"
                else:
                    rel_path = os.path.relpath(src, args.out)
        entities[e] = {"name": ent_name.get(e, str(e).replace("_", " ")), "image": rel_path}

    for i, tr in enumerate(triples):
        register_entity(tr["h"])
        register_entity(tr["t"])
        relations.setdefault(tr["r"], rel_name.get(tr["r"], str(tr["r"]).replace("_", " ")))
        out_triples.append({"id": f"t{i}", "h": tr["h"], "r": tr["r"], "t": tr["t"]})

    # gold supervision may reference (h,r,t) tuples instead of ids — build lookup
    hrt2id = {(x["h"], x["r"], x["t"]): x["id"] for x in out_triples}

    for split, path in zip(["train", "val", "test"], args.qa):
        with open(path) as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                raw = json.load(f)
            else:
                raw = [json.loads(l) for l in f if l.strip()]
        out_q = []
        for j, item in enumerate(raw):
            gold = pick(item, "gold_triplets", [])
            gold_ids = []
            for g in gold:
                if isinstance(g, (list, tuple)) and len(g) == 3:
                    gid = hrt2id.get(tuple(g))
                    if gid:
                        gold_ids.append(gid)
                else:
                    gold_ids.append(str(g))
            img = pick(item, "image")
            if img and args.image_root and not os.path.isabs(img):
                src = os.path.join(args.image_root, img)
                img = os.path.relpath(src, args.out) if os.path.exists(src) else None
            out_q.append({
                "id": str(pick(item, "id", f"{split}_{j}")),
                "text": pick(item, "text", ""),
                "image": img,
                "gold_triplets": gold_ids,
                "question": pick(item, "question", pick(item, "text", "")),
                "answers": pick(item, "answers", []),
            })
        with open(os.path.join(args.out, f"queries_{split}.json"), "w") as f:
            json.dump(out_q, f, indent=1)
        print(f"{split}: {len(out_q)} queries")

    json.dump(entities, open(os.path.join(args.out, "entities.json"), "w"), indent=1)
    json.dump(relations, open(os.path.join(args.out, "relations.json"), "w"), indent=1)
    json.dump(out_triples, open(os.path.join(args.out, "triplets.json"), "w"), indent=1)
    print(f"Done: {len(entities)} entities, {len(relations)} relations, "
          f"{len(out_triples)} triplets -> {args.out}")


if __name__ == "__main__":
    main()
