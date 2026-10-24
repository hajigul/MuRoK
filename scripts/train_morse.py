"""Train MORSE on a dataset's training queries.

Usage (Bench-G):
  python scripts/train_morse.py --data data/bench_g --epochs 10 \
      --save outputs/morse_g.pt
FB15k-237-IMG:
  python scripts/train_morse.py --data data/fb15k237img --epochs 10 \
      --remine-every 3 --save results/morse_fb_v4.pt

The frozen encoders and cached index are reused; only the gate g_theta is
trained. The index now includes a 4th candidate view, e_cap (caption-
augmented text), built automatically from data/<dataset>/captions.json if
present (run scripts/build_captions.py first) — this powers MORSE's
caption expert. A 5th expert (pure image-image similarity) needs no extra
indexing; it's computed on the fly from the raw (un-fused) query image
embedding against e_img.

Hard negatives are mined once from the static-fusion ranking (top-M
non-gold); optionally re-mined from the current MORSE checkpoint every
`--remine-every` epochs once it starts beating fusion. In-batch random
negatives are added for regularization, and the model is optimized with a
listwise InfoNCE loss under a cosine LR schedule and an annealed routing
temperature (soft mixing early, closer to hard routing later). Validation
NDCG@5 (S4, falling back to S2) is reported each epoch; the best
checkpoint is kept.

NOTE: this trains the current 5-expert morse.MorseScorer/MorseGate
architecture (text, visual, structural, caption, image-image). Checkpoints
from an older gate shape are NOT compatible and must be retrained with
this script.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from morse.data import MKG, load_queries, filter_setting
from morse.encoders import FrozenCLIP
from morse.index import build_index
from morse.metrics import evaluate_retrieval
from morse.morse import MorseConfig, MorseRetriever, MorseScorer, save_checkpoint
from morse.retriever import make_baseline


def mine_hard_negatives(retriever, train_q, gold_sets, depth, top_n, batch_size=128):
    """Rank train_q with `retriever` and keep top non-gold candidates per query."""
    ranked = retriever.retrieve_ranked(train_q, None, depth=depth, batch_size=batch_size)
    return [[j for j in h if j not in g][:top_n] for h, g in zip(ranked, gold_sets)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="floor for the cosine LR schedule")
    ap.add_argument("--hard-negatives", type=int, default=32)
    ap.add_argument("--rand-negatives", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--disable-caption", action="store_true",
                    help="ablation: permanently mask the caption expert "
                         "(force-off, regardless of image/caption "
                         "availability) so the router never uses it")
    ap.add_argument("--disable-image-image", action="store_true",
                    help="ablation: permanently mask the image-image expert")
    ap.add_argument("--disable-visual", action="store_true",
                    help="ablation: permanently mask the (cross-modal) "
                         "visual expert")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="InfoNCE temperature")
    ap.add_argument("--temp-start", type=float, default=2.0,
                    help="routing softmax temperature at epoch 1 (soft mixing)")
    ap.add_argument("--temp-end", type=float, default=0.7,
                    help="routing softmax temperature at the final epoch "
                         "(closer to hard expert selection)")
    ap.add_argument("--remine-every", type=int, default=0,
                    help="if >0, re-mine hard negatives from the CURRENT "
                         "MORSE checkpoint every N epochs instead of only "
                         "using the static-fusion mining done once up front "
                         "(0 disables re-mining)")
    ap.add_argument("--modality-dropout", type=float, default=0.0,
                    help="EMPIRICALLY DISCOURAGED as of the FB15k-237-IMG "
                         "experiments: even annealed to 0 by the final "
                         "epoch, corrupting a fraction of multimodal "
                         "training examples into fake text-only ones "
                         "consistently hurt the majority (image-rich) "
                         "settings without cleanly fixing the minority "
                         "(text-only) ones. Kept only for comparison; "
                         "prefer --text-only-augment instead, which adds "
                         "signal rather than removing/corrupting it. "
                         "0 disables (default, recommended).")
    ap.add_argument("--max-train", type=int, default=20000,
                    help="cap training queries for speed")
    ap.add_argument("--text-only-oversample", type=int, default=1,
                    help="duplicate naturally text-only training queries "
                         "(no resolvable image) this many times in the "
                         "training pool. Purely additive: does not touch "
                         "any multimodal training example. 1 disables.")
    ap.add_argument("--text-only-augment", type=int, default=0,
                    help="ADD this many synthetic text-only training "
                         "instances, built by taking randomly sampled "
                         "MULTIMODAL training queries and forcing them "
                         "(qbar -> pure-text embedding, q_img zeroed, "
                         "delta_q=0) to look exactly like a real text-only "
                         "query would -- same gold triplet, same relation "
                         "distribution as the real multimodal pool, just "
                         "presented without the image. These are APPENDED "
                         "to the training pool; no existing multimodal "
                         "example is removed, dropped, or altered, so this "
                         "cannot damage majority-regime training the way "
                         "--modality-dropout did. 0 disables.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    mkg = MKG.load(args.data)
    clip = FrozenCLIP(device=device)
    idx = build_index(mkg, clip, m=8, data_root=args.data)
    print(f"index: {int(idx.has_img.sum())}/{len(idx.has_img)} candidates have an image, "
          f"{int(idx.has_caption.sum())}/{len(idx.has_caption)} have a caption "
          f"(caption expert {'ACTIVE' if idx.has_caption.any() else 'inactive: no captions.json / build_captions.py not run yet'})")
    dim = idx.e_txt.shape[1]

    train_q = [q for q in load_queries(args.data, "train") if q.gold_triplets]
    rng.shuffle(train_q)
    train_q = train_q[: args.max_train]
    force_text = [False] * len(train_q)  # parallel marker: True = present as text-only at caching time

    if args.text_only_oversample > 1:
        text_only = [q for q in train_q if not q.has_visual_evidence]
        train_q = train_q + text_only * (args.text_only_oversample - 1)
        force_text = force_text + [False] * (len(text_only) * (args.text_only_oversample - 1))
        print(f"text-only oversampling: {len(text_only)} native text-only "
              f"queries duplicated {args.text_only_oversample}x "
              f"-> {len(text_only) * args.text_only_oversample} instances")

    if args.text_only_augment > 0:
        mm_pool = [q for q in train_q if q.has_visual_evidence]
        synth = [rng.choice(mm_pool) for _ in range(args.text_only_augment)] if mm_pool else []
        train_q = train_q + synth
        force_text = force_text + [True] * len(synth)
        print(f"text-only augmentation: +{len(synth)} synthetic text-only "
              f"instances (sampled with replacement from the {len(mm_pool)} "
              f"multimodal training queries, gold triplets unchanged, "
              f"presented without their image) -- ADDED on top of, not "
              f"instead of, the original multimodal training examples")

    order_shuffle = list(range(len(train_q)))
    rng.shuffle(order_shuffle)
    train_q = [train_q[i] for i in order_shuffle]
    force_text = [force_text[i] for i in order_shuffle]

    val_q = load_queries(args.data, "val")
    print(f"train queries (after any oversample/augment): {len(train_q)}")
    n_img_ok = sum(q.has_visual_evidence and not f for q, f in zip(train_q, force_text))
    print(f"train instances presented as multimodal: {n_img_ok}/{len(train_q)}  "
          f"(remaining {len(train_q) - n_img_ok} presented as text-only, "
          f"either natively or via --text-only-augment)")

    gold_sets = [set(mkg.tid2idx[t] for t in q.gold_triplets if t in mkg.tid2idx)
                 for q in train_q]

    # ---- mine hard negatives once with the static fusion baseline --------
    fusion = make_baseline("fusion", mkg, idx, clip)
    print("mining hard negatives with fusion top-100 ...")
    hard = mine_hard_negatives(fusion, train_q, gold_sets, depth=100,
                               top_n=args.hard_negatives)

    # ---- cache query embeddings + delta_q + raw query-image embeddings ----
    print("embedding training queries (one-time, frozen) ...")
    qbar_all, delta_q_all, qimg_all, qtext_all = [], [], [], []
    helper = MorseRetriever(mkg, idx, clip, MorseScorer(dim))
    for i in tqdm(range(0, len(train_q), 128)):
        qb, dq, qi, qt = helper.embed_queries_with_delta(train_q[i: i + 128])
        qbar_all.append(qb)
        delta_q_all.append(dq)
        qimg_all.append(qi)
        qtext_all.append(qt)
    qbar_all = torch.cat(qbar_all).cpu()
    delta_q_all = torch.cat(delta_q_all).cpu()
    qimg_all = torch.cat(qimg_all).cpu()
    qtext_all = torch.cat(qtext_all).cpu()

    force_text_t = torch.tensor(force_text, dtype=torch.bool)
    if force_text_t.any():
        qbar_all[force_text_t] = qtext_all[force_text_t]
        qimg_all[force_text_t] = 0.0
        delta_q_all[force_text_t] = 0.0

    scorer = MorseScorer(dim, MorseConfig(hidden=args.hidden,
                                          temperature=args.temp_start,
                                          disable_caption=args.disable_caption,
                                          disable_image_image=args.disable_image_image,
                                          disable_visual=args.disable_visual)).to(device)
    n_par = sum(p.numel() for p in scorer.parameters())
    print(f"trainable parameters: {n_par}")
    opt = torch.optim.AdamW(scorer.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    e_txt = idx.e_txt.to(device); e_img = idx.e_img.to(device)
    e_ctx = idx.e_ctx.to(device); e_cap = idx.e_cap.to(device)
    e_rel = idx.e_rel.to(device)
    has_img = idx.has_img.to(device); has_caption = idx.has_caption.to(device)
    n_cand = e_txt.shape[0]

    def temp_for_epoch(ep: int) -> float:
        """Linear anneal from temp_start (soft mixing) to temp_end (near-hard
        routing) across training, so early epochs get easier gradients and
        later epochs commit to confident per-relation routing."""
        if args.epochs <= 1:
            return args.temp_end
        frac = ep / (args.epochs - 1)
        return args.temp_start + frac * (args.temp_end - args.temp_start)

    def dropout_for_epoch(ep: int) -> float:
        """Linear anneal modality dropout from --modality-dropout down to 0
        across training. Applying it early nudges the router toward
        balanced routing while there's still plenty of optimization left to
        recover; fading it to 0 means the FINAL epochs (which set the
        checkpoint most epochs end up close to) train on the true,
        uncorrupted data distribution rather than a permanently distorted
        one -- avoiding the majority-regime damage seen with a flat rate."""
        if args.epochs <= 1:
            return 0.0
        frac = ep / (args.epochs - 1)
        return args.modality_dropout * (1.0 - frac)

    best = -1.0
    order = list(range(len(train_q)))
    for ep in range(args.epochs):
        scorer.cfg.temperature = temp_for_epoch(ep)
        cur_dropout = dropout_for_epoch(ep)
        scorer.train()
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for s0 in tqdm(range(0, len(order), args.batch_size),
                       desc=f"epoch {ep+1} (T={scorer.cfg.temperature:.2f}, "
                            f"drop={cur_dropout:.2f}, "
                            f"lr={sched.get_last_lr()[0]:.2e})"):
            ids = order[s0: s0 + args.batch_size]
            cand_rows, pos_col = [], []
            for qi in ids:
                pos = rng.choice(sorted(gold_sets[qi]))
                negs = hard[qi] + [rng.randrange(n_cand)
                                   for _ in range(args.rand_negatives)]
                negs = [n for n in negs if n != pos]
                cand_rows.append([pos] + negs)
                pos_col.append(0)
            width = min(len(r) for r in cand_rows)
            cand = torch.tensor([r[:width] for r in cand_rows], device=device)
            qbar = qbar_all[ids].clone().to(device)
            delta_q = delta_q_all[ids].clone().to(device)
            q_img = qimg_all[ids].clone().to(device)
            if cur_dropout > 0:
                is_mm = delta_q > 0.5
                drop = (torch.rand(len(ids), device=device) < cur_dropout) & is_mm
                if drop.any():
                    qbar[drop] = qtext_all[ids].to(device)[drop]
                    q_img[drop] = 0.0
                    delta_q[drop] = 0.0
            s = scorer(qbar, e_txt[cand], e_img[cand], e_ctx[cand],
                       e_cap[cand], e_rel[cand], has_img[cand],
                       has_caption[cand], delta_q, q_img)
            loss = F.cross_entropy(s / args.tau,
                                   torch.tensor(pos_col, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        # ---- validation NDCG@5 on the multimodal setting ------------------
        scorer.eval()
        retr = MorseRetriever(mkg, idx, clip, scorer)
        sq, mask = filter_setting(val_q, mkg, "S4")
        if not sq:
            sq, mask = filter_setting(val_q, mkg, "S2")
        gold = [set(mkg.tid2idx[t] for t in q.gold_triplets) for q in sq]
        m = evaluate_retrieval(retr.retrieve_ranked(sq, mask, depth=100), gold)
        print(f"epoch {ep+1}: loss {tot/max(nb,1):.4f} | "
              f"val NDCG@5 {m['NDCG@5']:.2f}")
        if m["NDCG@5"] > best:
            best = m["NDCG@5"]
            save_checkpoint(scorer.cpu(), dim, args.save)
            scorer.to(device)
            print(f"  saved best -> {args.save}")

        # ---- optional iterative hard-negative re-mining -------------------
        if args.remine_every and (ep + 1) % args.remine_every == 0 and ep + 1 < args.epochs:
            print("  re-mining hard negatives from current MORSE checkpoint ...")
            retr.scorer.eval()
            hard = mine_hard_negatives(retr, train_q, gold_sets, depth=100,
                                       top_n=args.hard_negatives)

    print(f"done. best val NDCG@5 = {best:.2f}")


if __name__ == "__main__":
    main()
