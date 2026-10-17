"""Load the supplied ICD-10-CM and SNOMED CT dictionaries.

The xlsx files are slow to parse, so the first load converts them to parquet
under ``artifacts/`` and later loads read the parquet copy.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .paths import ARTIFACTS, RAW

ICD_XLSX = RAW / "icd10cm_codes.xlsx"
SNOMED_XLSX = RAW / "SNOMEDCT_US_codes_and_descriptions.xlsx"

_SEMTAG = re.compile(r"\s*\(([a-z /\-]+)\)\s*$")
SEMTAGS = {
    "disorder", "finding", "procedure", "body structure", "organism", "substance",
    "physical object", "qualifier value", "observable entity", "medicinal product",
    "clinical drug", "medicinal product form", "situation", "morphologic abnormality",
    "event", "regime/therapy", "occupation", "specimen", "assessment scale", "product",
    "environment", "attribute", "cell", "geographic location", "navigational concept",
    "dose form", "cell structure", "record artifact", "person", "namespace concept",
    "disposition", "foundation metadata concept", "religion/philosophy", "physical force",
    "tumor staging", "role", "core metadata concept", "basic dose form", "social concept",
    "unit of presentation", "staging scale", "life style", "ethnic group", "racial group",
    "environment / location", "inactive concept", "administration method", "property",
    "release characteristic", "transformation", "state of matter", "intended site",
    "number", "supplier", "link assertion", "linkage concept", "special concept",
    "context-dependent category", "metadata", "observable", "regimen/therapy",
    "product name", "disposition", "role", "dose form",
}


def _cache(name: str) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / f"{name}.parquet"


def load_icd() -> pd.DataFrame:
    """Rows: code, description, billable (leaf), is_range."""
    p = _cache("icd_dict")
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_excel(ICD_XLSX, dtype=str).rename(columns={"Code": "code", "Description": "description"})
    df = df.dropna(subset=["code", "description"])
    df["code"] = df["code"].str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["is_range"] = df["code"].str.contains("-")
    codes = sorted(set(df.loc[~df["is_range"], "code"]))
    # a code is a leaf (billable) when no other code extends it
    leaf = {}
    for i, c in enumerate(codes):
        nxt = codes[i + 1] if i + 1 < len(codes) else ""
        leaf[c] = not nxt.startswith(c)
    df["billable"] = df["code"].map(leaf).fillna(False)
    df = df.drop_duplicates(["code", "description"]).reset_index(drop=True)
    df.to_parquet(p)
    return df


def load_snomed() -> pd.DataFrame:
    """Rows: code, description, order, semtag (from the FSN, propagated per code)."""
    p = _cache("snomed_dict")
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_excel(SNOMED_XLSX, sheet_name="SNOMED codes", dtype={"snomed_code": str})
    df = df.rename(columns={"snomed_code": "code", "description_order": "order"})
    df = df.dropna(subset=["code", "description"])
    df["code"] = df["code"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    tag = df["description"].str.extract(_SEMTAG)[0]
    df["semtag"] = tag.where(tag.isin(SEMTAGS))
    code_tag = df.dropna(subset=["semtag"]).groupby("code")["semtag"].first()
    df["semtag"] = df["code"].map(code_tag).fillna("unknown")
    df = df.drop_duplicates(["code", "description"]).reset_index(drop=True)
    df.to_parquet(p)
    return df
