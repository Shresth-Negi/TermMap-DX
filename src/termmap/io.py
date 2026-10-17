"""Input/output schema handling.

Training file columns: entity, chunk_text, coding_system, code, display, subconcept.
Evaluation input may use the training layout (without code/display) or the
brief's layout (chunk_id, clinical_entity[, chunk_text]). Column names are
normalised here so the engine accepts either.
"""
from __future__ import annotations

import pandas as pd

from .paths import TRAIN_CSV

ALIASES = {
    "clinical_entity": "entity", "term": "entity",
    "chunk": "chunk_text", "context": "chunk_text", "chunk_context": "chunk_text",
    "system": "coding_system",
}


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    return df


def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.rename(columns={c: ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns})
    if "entity" not in df:
        raise ValueError(f"input needs an entity/clinical_entity column, got {list(df.columns)}")
    if "chunk_text" not in df:
        df["chunk_text"] = ""
    if "subconcept" not in df:
        df["subconcept"] = ""
    return df
