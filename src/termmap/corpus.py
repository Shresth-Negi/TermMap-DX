"""Retrieval corpora: one row per unique normalised description, with the codes it names."""
from __future__ import annotations

import re

import pandas as pd

from .dictionaries import _ICD_ABBR, load_icd, load_snomed
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
        d = d[~d["is_range"]][["code", "description", "billable", "official"]].copy()
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


_TITLE_KW = re.compile(r"\b(other|unspecified|due to|with|without|not elsewhere classified|classified elsewhere)\b|, ",
                       re.I)


def _icd_title_score(desc: str, pos: int) -> float:
    """Display-title rule for ICD rows: the first row is usually the official title unless it is
    an abbreviated short title, an NOS index term or lowercase; official titles tend to carry
    'other' / 'unspecified' / 'with' / ', ' phrasing (85% agreement with the gold display)."""
    s = 0.0
    if desc.endswith("NOS") or " NOS" in desc:
        s -= 10
    if _ICD_ABBR.search(desc):
        s -= 5
    if not desc[:1].isupper():
        s -= 8
    if pos == 0:
        s += 1.0
    if _TITLE_KW.search(desc):
        s += 1.5
    return s + min(len(desc), 120) / 1000


def description_map(system: str, model_input: bool = False) -> dict[str, str]:
    """code -> title.

    model_input=False: human-readable title for outputs (SNOMED preferred term; ICD display rule).
    model_input=True: the text the cross-encoder / LLM were trained on (SNOMED preferred term
    plus semantic tag, e.g. 'Chest pain [finding]'; ICD 'official' column)."""
    e = load_entries(system)
    if system == "SNOMEDCT_US":
        e = e.sort_values("order", kind="stable").drop_duplicates("code").set_index("code")
        if model_input:
            return {c: f"{d} [{t}]" if t != "unknown" else d for c, d, t in zip(e.index, e["description"], e["semtag"])}
        return e["description"].to_dict()
    if model_input:
        e = e.sort_values("official", ascending=False, kind="stable")
        return e.drop_duplicates("code").set_index("code")["description"].to_dict()
    out = {}
    for code, g in e.groupby("code", sort=False):
        descs = g["description"].tolist()
        out[code] = max(enumerate(descs), key=lambda t: _icd_title_score(t[1], t[0]))[1]
    return out
