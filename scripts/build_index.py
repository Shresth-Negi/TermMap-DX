"""Embed every unique dictionary description with the given encoder(s).

Usage: python scripts/build_index.py --models sapbert gte --systems ICD10CM SNOMEDCT_US
Outputs artifacts/emb_<model>_<system>.npy (float16, L2-normalised, row = desc_id).
"""
import argparse
import time

import numpy as np

from termmap.corpus import SYSTEMS, load_corpus
from termmap.encoders import Encoder, set_threads
from termmap.paths import ARTIFACTS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["sapbert", "gte"])
ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
ap.add_argument("--limit", type=int, default=0, help="debug: only embed first N")
ap.add_argument("--batch", type=int, default=512)
args = ap.parse_args()
set_threads()

for m in args.models:
    enc = Encoder(m)
    for sys_ in args.systems:
        out = ARTIFACTS / f"emb_{m}_{sys_}.npy"
        if out.exists() and not args.limit:
            print("skip", out)
            continue
        texts = load_corpus(sys_)["norm"].tolist()
        if args.limit:
            texts = texts[: args.limit]
        t = time.time()
        v = enc.encode(texts, batch_size=args.batch, show=True)
        dt = time.time() - t
        print(f"{m} {sys_} {len(texts)} texts in {dt:.1f}s ({len(texts)/dt:.0f}/s) on {enc.device}", flush=True)
        if not args.limit:
            np.save(out, v.astype(np.float16))
