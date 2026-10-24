"""Generate a small synthetic multimodal KG in the expected on-disk format.

This exists so you can smoke-test the entire pipeline (indexing, MORSE
retrieval, all 5 settings, generation, ablations) in minutes before
plugging in the real MKG-RAG-Bench data. It creates simple colored-shape
images with PIL, entities/relations/triplets, and QA queries whose gold
triplets are recoverable from the graph.

Usage:
  python scripts/make_toy_data.py --out data/toy --n-entities 300 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random

from PIL import Image, ImageDraw

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
SHAPES = ["circle", "square", "triangle"]
RELATIONS = {
    "r_color": "has color",          # visually grounded
    "r_shape": "has shape",          # visually grounded
    "r_located": "located in",       # visually grounded-ish
    "r_type": "is a type of",        # lexical / taxonomic
    "r_related": "is related to",    # lexical
}
PLACES = ["forest", "desert", "ocean", "city", "mountain", "laboratory"]
TYPES = ["organism", "artifact", "structure", "vehicle", "instrument"]


def draw_image(path: str, color: str, shape: str) -> None:
    img = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([24, 24, 104, 104], fill=color)
    elif shape == "square":
        d.rectangle([24, 24, 104, 104], fill=color)
    else:
        d.polygon([(64, 20), (18, 108), (110, 108)], fill=color)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n-entities", type=int, default=300)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    entities, triplets = {}, []
    attr = {}
    for i in range(args.n_entities):
        eid = f"e{i}"
        color, shape = rng.choice(COLORS), rng.choice(SHAPES)
        has_img = rng.random() < 0.5
        img_rel = None
        if has_img:
            img_rel = f"images/{eid}.png"
            draw_image(os.path.join(args.out, img_rel), color, shape)
        entities[eid] = {"name": f"object {i}", "image": img_rel}
        attr[eid] = {"color": color, "shape": shape,
                     "place": rng.choice(PLACES), "type": rng.choice(TYPES)}

    # attribute entities (no images)
    for name in COLORS + SHAPES + PLACES + TYPES:
        entities[f"a_{name}"] = {"name": name, "image": None}

    tid = 0
    def add(h, r, t):
        nonlocal tid
        triplets.append({"id": f"t{tid}", "h": h, "r": r, "t": t})
        tid += 1

    for eid, a in attr.items():
        add(eid, "r_color", f"a_{a['color']}")
        add(eid, "r_shape", f"a_{a['shape']}")
        add(eid, "r_located", f"a_{a['place']}")
        add(eid, "r_type", f"a_{a['type']}")
        # a couple of lateral relations for graph density
        other = f"e{rng.randrange(args.n_entities)}"
        if other != eid:
            add(eid, "r_related", other)

    # gold lookup: (h, r) -> triplet id
    hr2tid = {(x["h"], x["r"]): x["id"] for x in triplets}

    queries = []
    obj_ids = list(attr.keys())
    for qi in range(args.n_queries):
        eid = rng.choice(obj_ids)
        a = attr[eid]
        rel = rng.choice(["r_color", "r_shape", "r_located", "r_type"])
        gold = [hr2tid[(eid, rel)]]
        ans = {"r_color": a["color"], "r_shape": a["shape"],
               "r_located": a["place"], "r_type": a["type"]}[rel]
        rel_name = RELATIONS[rel]
        # half the queries are multimodal (carry the entity image, if it has one)
        use_img = entities[eid]["image"] is not None and rng.random() < 0.7
        q = {
            "id": f"q{qi}",
            "text": f"What {rel_name.replace('has ', '')} does {entities[eid]['name']} have?"
                    if rel in ("r_color", "r_shape")
                    else f"Where is {entities[eid]['name']}?" if rel == "r_located"
                    else f"What type of thing is {entities[eid]['name']}?",
            "image": entities[eid]["image"] if use_img else None,
            "gold_triplets": gold,
            "question": None,
            "answers": [ans],
        }
        q["question"] = q["text"]
        queries.append(q)

    rng.shuffle(queries)
    n = len(queries)
    splits = {"train": queries[: int(0.7 * n)],
              "val": queries[int(0.7 * n): int(0.85 * n)],
              "test": queries[int(0.85 * n):]}

    with open(os.path.join(args.out, "entities.json"), "w") as f:
        json.dump(entities, f, indent=1)
    with open(os.path.join(args.out, "relations.json"), "w") as f:
        json.dump(RELATIONS, f, indent=1)
    with open(os.path.join(args.out, "triplets.json"), "w") as f:
        json.dump(triplets, f, indent=1)
    for split, qs in splits.items():
        with open(os.path.join(args.out, f"queries_{split}.json"), "w") as f:
            json.dump(qs, f, indent=1)

    print(f"Toy MKG written to {args.out}: "
          f"{len(entities)} entities, {len(triplets)} triplets, "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} train/val/test queries")


if __name__ == "__main__":
    main()
