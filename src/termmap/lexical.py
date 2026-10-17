"""Sparse lexical retrieval: character n-gram TF-IDF (robust to OCR noise and
misspellings) and word TF-IDF (rewards exact rare tokens)."""
from __future__ import annotations

import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .paths import ARTIFACTS

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None


def _vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True,
                               dtype=np.float32, min_df=1)
    return TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", sublinear_tf=True,
                           dtype=np.float32, min_df=1)


class LexicalIndex:
    def __init__(self, kind: str, system: str):
        self.kind, self.system = kind, system
        self.vec: TfidfVectorizer | None = None
        self.mat: sp.csr_matrix | None = None

    @property
    def path(self):
        return ARTIFACTS / f"lex_{self.kind}_{self.system}.pkl"

    def build(self, texts: list[str]) -> "LexicalIndex":
        self.vec = _vectorizer(self.kind)
        self.mat = self.vec.fit_transform(texts).tocsr()
        return self

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump((self.vec, self.mat), f, protocol=4)

    @classmethod
    def load_or_build(cls, kind: str, system: str, texts_fn) -> "LexicalIndex":
        ix = cls(kind, system)
        if ix.path.exists():
            with open(ix.path, "rb") as f:
                ix.vec, ix.mat = pickle.load(f)
            return ix
        ix.build(texts_fn())
        ix.save()
        return ix

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.vec.transform(texts).tocsr()

    def search(self, texts: list[str], k: int = 50, threads: int = 8):
        """Return (ids, scores) arrays of shape (n, k); missing slots are -1 / 0."""
        q = self.transform(texts)
        if sp_matmul_topn is not None:
            if getattr(self, "_matT", None) is None:
                self._matT = self.mat.T.tocsr()
            r = sp_matmul_topn(q, self._matT, top_n=k, threshold=1e-4, sort=True,
                               n_threads=threads).tocsr()
        else:
            r = (q @ self.mat.T).tocsr()
        ids = np.full((len(texts), k), -1, dtype=np.int64)
        sc = np.zeros((len(texts), k), dtype=np.float32)
        for i in range(len(texts)):
            s, e = r.indptr[i], r.indptr[i + 1]
            cols, vals = r.indices[s:e], r.data[s:e]
            o = np.argsort(-vals)[:k]
            ids[i, : len(o)] = cols[o]
            sc[i, : len(o)] = vals[o]
        return ids, sc

    def pair_scores(self, qmat: sp.csr_matrix, qrow: np.ndarray, doc_ids: np.ndarray) -> np.ndarray:
        """Cosine between query rows and specific docs (vectorised row-wise dot)."""
        a = qmat[qrow]
        b = self.mat[doc_ids]
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()
