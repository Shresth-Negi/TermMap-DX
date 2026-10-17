"""Retrieval corpora: one row per unique normalised description, with the codes it names."""
from __future__ import annotations

import pandas as pd

from .dictionaries import load_icd, load_snomed
from .normalize import basic_norm, strip_semtag
from .paths import ARTIFACTS

SYSTEMS = ("ICD10CM", "SNOMEDCT_US")


def load_entries(system: str) -> pd.DataFrame:
    """code, description, norm, desc_id. ICD chapter/block ranges are dropped (not codes)."""
    p = ARTIFACTS / f"entries_{system}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    if system == "ICD10CM":
        d = load_icd()
        d = d[~d["is_range"]][["code", "description", "billable"]].copy()
    else:
        d = load_snomed()[["code", "description", "order", "semtag"]].copy()
    d["norm"] = d["description"].map(basic_norm if system == "ICD10CM" else strip_semtag)
    d = d[d["norm"].str.len() > 0]
    uniq = pd.Index(d["norm"].unique())
    d["desc_id"] = uniq.get_indexer(d["norm"])
    d = d.reset_index(drop=True)
    d.to_parquet(p)
    return d


def load_corpus(system: str) -> pd.DataFrame:
    """desc_id-indexed frame: norm text and the list of codes sharing it."""
    p = ARTIFACTS / f"corpus_{system}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    e = load_entries(system)
    c = e.groupby("desc_id").agg(norm=("norm", "first"), codes=("code", lambda s: sorted(set(s))))
    c = c.sort_index()
    c.to_parquet(p)
    return c
