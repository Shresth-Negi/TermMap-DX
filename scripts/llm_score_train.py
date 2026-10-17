"""Score the out-of-fold stage-1 shortlists of the training set with a local LLM (GPU).

The LLM is zero-shot (never trained on our labels), so its scores can be used as
a stage-2 feature without leakage. Output: artifacts/llm_scores_train<tag>.parquet

Usage: python scripts/llm_score_train.py --tag _v1 [--model Qwen/Qwen2.5-7B-Instruct] [--limit N]
"""
import argparse
import time

import pandas as pd

from termmap.corpus import SYSTEMS, description_map
from termmap.encoders import Encoder
from termmap.features import QueryEncoder
from termmap.io import load_train
from termmap.llm_scorer import LLMScorer, neighbour_examples, score_shortlists
from termmap.pipeline import attach_keys, build_memory, make_keys
from termmap.paths import ARTIFACTS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out-suffix", default="", help="appended to the output file name")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--shots", type=int, default=5, help="nearest training examples shown in the prompt (0 = zero-shot)")
args = ap.parse_args()

keys = pd.read_parquet(ARTIFACTS / "train_keys.parquet").set_index("key")
tr = load_train()
tr["key"] = attach_keys(tr, make_keys(tr)).values
qenc = QueryEncoder({"sapbert": Encoder("sapbert")})
scorer = LLMScorer(args.model)
out = []
for sys_ in SYSTEMS:
    X = pd.read_parquet(ARTIFACTS / f"feats2_{sys_}{args.tag}.parquet", columns=["key", "code", "s1_rank"])
    if args.limit:
        X = X[X.key.isin(X.key.drop_duplicates().head(args.limit))]
    desc_of = description_map(sys_)
    mem, mem_keys = build_memory(tr, sys_, qenc, ("sapbert",))
    qk = keys.loc[X.key.unique()].reset_index()
    ex = neighbour_examples(mem, mem_keys, qenc, "sapbert", qk, desc_of, sys_, k=args.shots, loo=True) if args.shots else None
    t = time.time()
    r = score_shortlists(scorer, sys_, X, keys, desc_of, batch_size=args.batch, examples=ex)
    print(f"{sys_}: {r.key.nunique()} keys scored in {time.time()-t:.0f}s", flush=True)
    out.append(r)
pd.concat(out).to_parquet(ARTIFACTS / f"llm_scores_train{args.tag}{args.out_suffix}.parquet")
print("saved")
