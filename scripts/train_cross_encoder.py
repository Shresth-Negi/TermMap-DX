"""Fine-tune the shortlist cross-encoder: 5-fold out-of-fold scores + final model (GPU).

Reads the stage-2 shortlists written by train.py (feats2_<sys><tag>.parquet).
Writes artifacts/ce_scores_train<tag>.parquet and artifacts/ce_<sys><tag>/ (final models).

Usage: python scripts/train_cross_encoder.py --tag _d3 [--epochs 2]
"""
import argparse
import time

import pandas as pd

from termmap.corpus import SYSTEMS, description_map
from termmap.cross_encoder import CrossEncoder, build_groups
from termmap.paths import ARTIFACTS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_d3")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N shortlists")
ap.add_argument("--out-tag", default=None, help="tag for outputs (default: --tag)")
args = ap.parse_args()

OT = args.out_tag if args.out_tag is not None else args.tag
keys = pd.read_parquet(ARTIFACTS / "train_keys.parquet").set_index("key")
fold = keys["fold"]
out = []
for sys_ in args.systems:
    X = pd.read_parquet(ARTIFACTS / f"feats2_{sys_}{args.tag}.parquet", columns=["key", "code", "s1_rank", "label"])
    desc_of = description_map(sys_, with_semtag=True)
    groups = build_groups(sys_, X, keys, desc_of)
    if args.limit:
        groups = groups[: args.limit]
    print(f"[{sys_}] {len(groups)} shortlists, {sum(g[4] >= 0 for g in groups)} with gold", flush=True)
    for k in range(args.folds):
        t = time.time()
        trn = [g for g in groups if fold[g[0]] != k]
        val = [g for g in groups if fold[g[0]] == k]
        ce = CrossEncoder().fit(trn, epochs=args.epochs)
        s = ce.score(val)
        m = s.merge(X[["key", "code", "label"]], on=["key", "code"])
        m = m[m.groupby("key")["label"].transform("max") > 0]
        acc = m.sort_values("ce", ascending=False).drop_duplicates("key")["label"].mean()
        print(f"  fold {k}: {len(trn)} train / {len(val)} scored, top-1 within shortlist {acc:.4f} "
              f"({time.time()-t:.0f}s)", flush=True)
        out.append(s.assign(system=sys_))
        del ce
    final = CrossEncoder().fit(groups, epochs=args.epochs)
    final.save(str(ARTIFACTS / f"ce_{sys_}{OT}"))
pd.concat(out)[["system", "key", "code", "ce", "ce_lp"]].to_parquet(ARTIFACTS / f"ce_scores_train{OT}.parquet")
print("saved")
