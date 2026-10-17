"""Compare ranker settings on cached stage-1 features (same folds as train.py).

Usage: python scripts/tune_ranker.py --tag _sap
"""
import argparse
import json

import numpy as np
import pandas as pd

from termmap import ranker
from termmap.io import load_train
from termmap.paths import ARTIFACTS
from termmap.pipeline import attach_keys, make_keys

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_sap")
ap.add_argument("--systems", nargs="+", default=["ICD10CM", "SNOMEDCT_US"])
args = ap.parse_args()

tr = load_train()
keys = make_keys(tr)
tr["key"] = attach_keys(tr, keys).values
fk = pd.read_parquet(ARTIFACTS / "train_keys.parquet").set_index("key")["fold"]

VARIANTS = {
    "base": {},
    "lr02_l31": dict(learning_rate=0.02, num_leaves=31),
    "lr03_l127_mdl50": dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=50),
    "xendcg": dict(objective="rank_xendcg"),
    "binary": dict(objective="binary", metric="binary_logloss"),
    "ff05": dict(feature_fraction=0.5, lambda_l2=5.0),
}

out = {}
for sys_ in args.systems:
    F = pd.read_parquet(ARTIFACTS / f"feats1_{sys_}{args.tag}.parquet").reset_index(drop=True)
    g = tr[tr.coding_system == sys_].set_index("key")["code"]
    F["label"] = (F["code"].values == g.reindex(F["key"].values).values).astype(np.int8)
    F["labeled"] = F["key"].isin(g.index).values
    F["fold"] = fk.reindex(F["key"].values).values
    cols = ranker.feature_cols(F, drop=("key", "labeled"))
    for name, upd in VARIANTS.items():
        saved = dict(ranker.PARAMS)
        ranker.PARAMS.update(upd)
        oof = np.zeros(len(F))
        for k in range(5):
            trn, val = F[F.fold != k], F[F.fold == k]
            es = (k + 1) % 5
            bst = ranker.train_ranker(trn[(trn.fold != es) & trn.labeled], cols, rounds=3000,
                                      valid=trn[(trn.fold == es) & trn.labeled])
            oof[val.index.values] = bst.predict(val[cols])
        ranker.PARAMS.clear(); ranker.PARAMS.update(saved)
        d = F[["key", "code"]].assign(s=oof).sort_values(["key", "s"], ascending=[True, False]).drop_duplicates("key")
        pred = d.set_index("key")["code"].reindex(g.index)
        acc = float((pred.values == g.values).mean())
        out[f"{sys_}:{name}"] = round(acc, 4)
        print(sys_, name, round(acc, 4), flush=True)
print(json.dumps(out, indent=1))
