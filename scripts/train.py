"""Train the two-stage mapping engine and report cross-validated accuracy.

Stage 1: retrieval features -> LightGBM LambdaRank, 5-fold CV gives out-of-fold
         scores for every training key in both systems (folds are shared
         across systems, keyed by (entity, chunk, subconcept)).
Stage 2: shortlist of stage 1 + cross-system coupling + optional LLM listwise
         scores -> LightGBM LambdaRank, again 5-fold CV for the reported numbers.

Training-memory features are leave-one-out (a key never sees its own label).

Usage: python scripts/train.py [--dense sapbert gte] [--llm artifacts/llm_scores.parquet] [--tag NAME]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from termmap import ranker
from termmap.corpus import SYSTEMS
from termmap.encoders import Encoder, set_threads
from termmap.features import QueryEncoder
from termmap.io import load_train
from termmap.paths import ARTIFACTS
from termmap.pipeline import (Coupler, add_extra_features, attach_keys, bridge_variants, build_memory,
                              make_keys, shortlist, stage1_features, system_keys, top_probs)
from termmap.resources import SystemResources

ap = argparse.ArgumentParser()
ap.add_argument("--dense", nargs="*", default=["sapbert"])
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--llm", default="", help="parquet of LLM shortlist scores (system,key,code,llm_lp)")
ap.add_argument("--llm-model-id", default="Qwen/Qwen2.5-7B-Instruct", help="LLM that produced --llm (recorded for inference)")
ap.add_argument("--fallback-tag", default="",
                help="model set the engine uses instead when this one's LLM cannot run (no GPU / memory)")
ap.add_argument("--extra", nargs="*", default=[],
                help="more shortlist score files (system,key,code,<feature...>), e.g. cross-encoder scores")
ap.add_argument("--tag", default="")
ap.add_argument("--feats-tag", default=None, help="reuse cached stage-1 features of another tag")
ap.add_argument("--bridge", type=int, default=1, help="SNOMED-synonym query variants for ICD (1/0)")
args = ap.parse_args()
set_threads()
T = args.tag

tr = load_train()
keys = make_keys(tr)
tr["key"] = attach_keys(tr, keys).values
fold_of_key = np.zeros(len(keys), dtype=int)
for k, (_, te) in enumerate(KFold(args.folds, shuffle=True, random_state=7).split(np.arange(len(keys)))):
    fold_of_key[te] = k
keys[["key", "entity", "chunk_text", "subconcept"]].assign(fold=fold_of_key).to_parquet(ARTIFACTS / "train_keys.parquet")

(ARTIFACTS / f"config{T}.json").write_text(json.dumps({"dense": args.dense, "bridge": args.bridge,
                                                       "llm": bool(args.llm), "llm_model": args.llm_model_id if args.llm else "",
                                                       "extra": [Path(p).name for p in args.extra],
                                                       "fallback": args.fallback_tag}))
qenc = QueryEncoder({m: Encoder(m) for m in args.dense})
res, S, gold, report, acc1_of = {}, {}, {}, {}, {}


def cv_rank(F, cols, label_name):
    """Out-of-fold scores for every row of F + list of best iterations."""
    oof = np.zeros(len(F))
    iters = []
    for k in range(args.folds):
        trn, val = F[F.fold != k], F[F.fold == k]
        es = (k + 1) % args.folds
        bst = ranker.train_ranker(trn[(trn.fold != es) & trn.labeled], cols, rounds=2000,
                                  valid=trn[(trn.fold == es) & trn.labeled])
        iters.append(bst.best_iteration or 2000)
        oof[val.index.values] = bst.predict(val[cols], num_threads=ranker.PARAMS["num_threads"])
    print(f"  [{label_name}] best iterations {iters}", flush=True)
    return oof, iters


def accuracy(F, score, sys_):
    d = F[["key", "code"]].assign(s=score).sort_values(["key", "s", "code"], ascending=[True, False, True])
    top = d.drop_duplicates("key").set_index("key")["code"]
    g = gold[sys_]
    pred = top.reindex(g.index)
    return pred, float((pred.values == g.values).mean())


for sys_ in SYSTEMS:
    t0 = time.time()
    res[sys_] = SystemResources(sys_, dense_models=tuple(args.dense))
    g = tr[tr.coding_system == sys_].set_index("key")["code"]
    gold[sys_] = g
    fpath = ARTIFACTS / f"feats1_{sys_}{args.feats_tag if args.feats_tag is not None else T}.parquet"
    if fpath.exists() and not args.rebuild:
        F = pd.read_parquet(fpath)
    else:
        mem, mem_keys = build_memory(tr, sys_, qenc, tuple(args.dense))
        qk = system_keys(keys, sys_)
        extra = None
        if sys_ == "ICD10CM" and args.bridge and args.dense:
            res_sn = SystemResources("SNOMEDCT_US", dense_models=(args.dense[0],))
            extra = bridge_variants(res_sn, qenc, qk["entity"].tolist(), model=args.dense[0])
            del res_sn
        F = stage1_features(res[sys_], mem, mem_keys, qenc, qk, loo=True, extra_variants=extra)
        F.to_parquet(fpath)
    F = F.reset_index(drop=True)
    F["label"] = (F["code"].values == g.reindex(F["key"].values).values).astype(np.int8)
    F["labeled"] = F["key"].isin(g.index).values
    F["fold"] = fold_of_key[F["key"].values]
    cols1 = ranker.feature_cols(F, drop=("key", "labeled"))
    print(f"[{sys_}] stage-1 features {F.shape} ({time.time()-t0:.0f}s)", flush=True)
    FT = args.feats_tag if args.feats_tag is not None else T
    cache = ARTIFACTS / f"oof1_{sys_}{FT}.npy"
    reuse = args.feats_tag is not None and cache.exists() and (ARTIFACTS / f"ranker_{sys_}_s1{FT}.txt").exists()
    if reuse:  # stage 1 is deterministic given the features: reuse its OOF scores and model
        oof1, it1 = np.load(cache), None
        print(f"  [{sys_} s1] reusing cached stage-1 from tag {FT}", flush=True)
    else:
        oof1, it1 = cv_rank(F, cols1, f"{sys_} s1")
        np.save(ARTIFACTS / f"oof1_{sys_}{T}.npy", oof1)
    pred1, acc1 = accuracy(F, oof1, sys_)
    rec = F[F.labeled].groupby("key")["label"].max().reindex(g.index).fillna(0).mean()
    print(f"[{sys_}] stage-1 recall {rec:.4f}  CV top-1 {acc1:.4f}", flush=True)
    if reuse:
        bst, _ = ranker.load(sys_, "_s1" + FT)
    else:
        bst = ranker.train_ranker(F[F.labeled], cols1, rounds=int(np.median(it1) * 1.1))
    ranker.save(bst, cols1, sys_, "_s1" + T)
    S[sys_] = shortlist(F, oof1)
    report[sys_] = {"stage1_cv_top1": round(acc1, 4), "candidate_recall": round(float(rec), 4)}
    acc1_of[sys_] = acc1

# ------------------------------------------------------------------ stage 2
coupler = Coupler(tr, res)
final_stage = {}
extras = [pd.read_parquet(p) for p in ([args.llm] if args.llm else []) + args.extra]
for sys_ in SYSTEMS:
    other = "SNOMEDCT_US" if sys_ == "ICD10CM" else "ICD10CM"
    X = S[sys_].reset_index(drop=True)
    X = pd.concat([X, coupler.features(sys_, X, top_probs(S[other]))], axis=1)
    X = add_extra_features(X, [e[e.system == sys_] for e in extras])
    X = X.sort_values(["qid", "s1_rank"]).reset_index(drop=True)
    X.to_parquet(ARTIFACTS / f"feats2_{sys_}{T}.parquet")
    cols2 = ranker.feature_cols(X, drop=("key", "labeled"))
    oof2, it2 = cv_rank(X, cols2, f"{sys_} s2")
    pred2, acc2 = accuracy(X, oof2, sys_)
    sl_rec = X[X.labeled].groupby("key")["label"].max().reindex(gold[sys_].index).fillna(0).mean()
    print(f"[{sys_}] shortlist recall {sl_rec:.4f}  stage-2 CV top-1 {acc2:.4f}", flush=True)
    bst = ranker.train_ranker(X[X.labeled], cols2, rounds=int(np.median(it2) * 1.1))
    ranker.save(bst, cols2, sys_, "_s2" + T)
    imp = pd.Series(bst.feature_importance("gain"), index=cols2).sort_values(ascending=False)
    print(imp.head(20).round(0).to_string())
    # model selection on CV: keep stage 2 only if it beats stage 1 out of fold
    acc1 = acc1_of[sys_]
    use_s2 = acc2 > acc1
    final_stage[sys_] = 2 if use_s2 else 1
    fscore = oof2 if use_s2 else X["s1"].values
    pred_f, acc_f = (pred2, acc2) if use_s2 else accuracy(X, X["s1"].values, sys_)
    print(f"[{sys_}] final stage = {final_stage[sys_]}  (CV top-1 {acc_f:.4f})", flush=True)
    o = tr[tr.coding_system == sys_].set_index("key")[["entity", "chunk_text", "subconcept", "code"]].copy()
    o["pred"] = pred_f.reindex(o.index).values
    o["pred_s1"] = S[sys_].query("s1_rank == 0").set_index("key")["code"].reindex(o.index).values
    o["ok"] = o["pred"] == o["code"]
    o["in_shortlist"] = X[X.labeled].groupby("key")["label"].max().reindex(o.index).fillna(0).astype(bool)
    X2 = X.assign(s2=fscore).sort_values(["key", "s2"], ascending=[True, False])
    top = X2.groupby("key").head(2)
    marg = top.groupby("key")["s2"].agg(lambda s: s.iloc[0] - (s.iloc[1] if len(s) > 1 else s.iloc[0] - 10))
    o["margin"] = marg.reindex(o.index).values
    o["fold"] = fold_of_key[o.index.values]
    o.reset_index().to_parquet(ARTIFACTS / f"oof_{sys_}{T}.parquet")
    # calibration: P(correct | stage-2 margin), isotonic on out-of-fold predictions
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(o["margin"].fillna(0), o["ok"].astype(float))
    grid = np.linspace(np.nanpercentile(o["margin"], 0.5), np.nanpercentile(o["margin"], 99.5), 200)
    (ARTIFACTS / f"calib_{sys_}{T}.json").write_text(json.dumps({"x": grid.tolist(), "y": iso.predict(grid).tolist()}))
    report[sys_].update({"shortlist_recall": round(float(sl_rec), 4), "stage2_cv_top1": round(acc2, 4),
                         "final_stage": final_stage[sys_], "final_cv_top1": round(acc_f, 4),
                         "by_subconcept": o.groupby("subconcept")["ok"].mean().round(4).to_dict()})

report["headline_min_top1"] = min(report[s]["final_cv_top1"] for s in SYSTEMS)
cfg = json.loads((ARTIFACTS / f"config{T}.json").read_text())
cfg["final_stage"] = final_stage
(ARTIFACTS / f"config{T}.json").write_text(json.dumps(cfg))
(ARTIFACTS / f"cv_report{T}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
