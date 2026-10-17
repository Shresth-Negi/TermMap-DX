"""Fallback subconcept classifier (DiagnosisCode / ProcedureCode / LabService / ...).

Only used when an input file has no subconcept column, so the ranker still gets
its entity-type signal. TF-IDF over entity + local context -> logistic regression.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from .features import context_window
from .normalize import basic_norm
from .paths import ARTIFACTS

PATH = ARTIFACTS / "subconcept_clf.pkl"


def _text(entity: str, chunk: str) -> str:
    return basic_norm(entity) + " || " + context_window(entity, chunk, 40)


def fit(train: pd.DataFrame):
    k = train.drop_duplicates(["entity", "chunk_text", "subconcept"])
    X = [_text(e, c) for e, c in zip(k.entity, k.chunk_text)]
    clf = make_pipeline(
        make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2),
                   TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        LogisticRegression(C=5.0, max_iter=2000, random_state=0))
    clf.fit(X, k.subconcept.to_numpy(dtype=object))
    with open(PATH, "wb") as f:
        pickle.dump(clf, f)
    return clf


def predict(df: pd.DataFrame) -> np.ndarray:
    with open(PATH, "rb") as f:
        clf = pickle.load(f)
    return clf.predict([_text(e, c) for e, c in zip(df.entity, df.chunk_text)])
