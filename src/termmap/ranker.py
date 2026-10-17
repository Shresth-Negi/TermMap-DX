"""LightGBM LambdaRank over candidate features (one model per coding system)."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .paths import ARTIFACTS

META_COLS = {"qid", "code", "label", "src", "fold"}
CATEGORICAL = ["semtag", "subconcept", "chapter", "enc7", "cand_lat", "q_lat"]

PARAMS = dict(
    objective="lambdarank", metric="map", eval_at=[1], learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambdarank_truncation_level=20, lambda_l2=1.0, verbose=-1, seed=13, deterministic=True,
    force_col_wise=True, num_threads=16,
)


def feature_cols(df: pd.DataFrame, drop: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c not in drop]


def _dataset(df: pd.DataFrame, cols: list[str]):
    df = df.sort_values(["qid"], kind="stable")
    grp = df.groupby("qid", sort=False).size().values
    cats = [c for c in CATEGORICAL if c in cols]
    return lgb.Dataset(df[cols], label=df["label"].values, group=grp, categorical_feature=cats,
                       free_raw_data=False)


def train_ranker(df: pd.DataFrame, cols: list[str], rounds: int = 600, valid: pd.DataFrame | None = None):
    pos_q = df.groupby("qid")["label"].transform("max") > 0  # queries without a positive carry no signal
    dtr = _dataset(df[pos_q], cols)
    params = dict(PARAMS)
    if valid is not None:
        vq = valid.groupby("qid")["label"].transform("max") > 0
        dva = _dataset(valid[vq], cols)
        bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
    else:
        bst = lgb.train(params, dtr, num_boost_round=rounds)
    return bst


def predict_top(bst, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return per-qid best code, its score, margin over runner-up and softmax confidence."""
    d = df[["qid", "code"]].copy()
    d["score"] = bst.predict(df[cols], num_threads=PARAMS["num_threads"])
    d = d.sort_values(["qid", "score", "code"], ascending=[True, False, True], kind="stable")
    d["rank"] = d.groupby("qid").cumcount()
    d["p"] = np.exp(d["score"] - d.groupby("qid")["score"].transform("max"))
    d["p"] = d["p"] / d.groupby("qid")["p"].transform("sum")
    top = d[d["rank"] == 0].set_index("qid")
    second = d[d["rank"] == 1].set_index("qid")["score"]
    top["margin"] = top["score"] - second.reindex(top.index).fillna(top["score"] - 10)
    alts = d[d["rank"] < 5].groupby("qid")["code"].apply(list)
    top["top5"] = alts.reindex(top.index)
    return top[["code", "score", "p", "margin", "top5"]]


def save(bst, cols, system: str, tag: str = ""):
    bst.save_model(str(ARTIFACTS / f"ranker_{system}{tag}.txt"))
    (ARTIFACTS / f"ranker_{system}{tag}.json").write_text(json.dumps({"cols": cols}))


def load(system: str, tag: str = ""):
    bst = lgb.Booster(model_file=str(ARTIFACTS / f"ranker_{system}{tag}.txt"))
    cols = json.loads((ARTIFACTS / f"ranker_{system}{tag}.json").read_text())["cols"]
    return bst, cols
