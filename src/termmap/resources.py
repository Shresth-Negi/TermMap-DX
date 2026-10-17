"""Per-system lookup resources shared by candidate generation and features."""
from __future__ import annotations

from collections import defaultdict
from functools import cached_property

import numpy as np
import pandas as pd

from .corpus import load_corpus, load_entries
from .lexical import LexicalIndex
from .normalize import key_norm
from .paths import ARTIFACTS

MAX_DESC_PER_CODE = 40


class SystemResources:
    """Dictionary-side resources for one coding system (independent of training labels)."""

    def __init__(self, system: str, dense_models: tuple[str, ...] = ("sapbert",)):
        self.system = system
        self.corpus = load_corpus(system)
        self.entries = load_entries(system)
        self.texts = self.corpus["norm"].tolist()
        self.desc_codes = self.corpus["codes"].tolist()
        self.char = LexicalIndex.load_or_build("char", system, lambda: self.texts)
        self.word = LexicalIndex.load_or_build("word", system, lambda: self.texts)
        self.dense: dict[str, np.ndarray] = {}
        for m in dense_models:
            p = ARTIFACTS / f"emb_{m}_{system}.npy"
            if p.exists():
                self.dense[m] = np.load(p).astype(np.float32)
        self._build_code_tables()

    def _build_code_tables(self):
        e = self.entries
        self.codes = np.array(sorted(e["code"].unique()))
        self.code_idx = {c: i for i, c in enumerate(self.codes)}
        descs = defaultdict(list)
        for c, d in zip(e["code"], e["desc_id"]):
            lst = descs[c]
            if len(lst) < MAX_DESC_PER_CODE and d not in lst:
                lst.append(d)
        self.code_descs = {c: np.array(v, dtype=np.int64) for c, v in descs.items()}
        if self.system == "SNOMEDCT_US":
            first = e.sort_values("order").drop_duplicates("code")
            self.pref_desc = dict(zip(first["code"], first["desc_id"]))
            self.semtag = dict(zip(first["code"], first["semtag"]))
            self.billable = {}
        else:
            first = e.sort_values("official", ascending=False, kind="stable").drop_duplicates("code")
            self.pref_desc = dict(zip(first["code"], first["desc_id"]))
            self.billable = dict(zip(first["code"], first["billable"]))
            self.semtag = {}
            txt = e.assign(t=e["norm"]).groupby("code")["t"].agg(list)
            self.any_unspec = {c: any("unspecified" in x for x in v) for c, v in txt.items()}
            self.any_other = {c: any(x.startswith("other") or " other " in x for x in v) for c, v in txt.items()}
        self.desc_order = {}
        if "order" in e:
            for c, d, o in zip(e["code"], e["desc_id"], e["order"]):
                k = (c, d)
                if k not in self.desc_order or o < self.desc_order[k]:
                    self.desc_order[k] = int(o)
        keys = defaultdict(set)
        for c, n in zip(e["code"], e["norm"]):
            keys[key_norm(n)].add(c)
        self.key2codes = dict(keys)

    @cached_property
    def code_set(self) -> set[str]:
        return set(self.codes.tolist())

    def dense_search(self, model: str, q: np.ndarray, k: int = 50, chunk: int = 1024):
        D = self.dense[model]
        ids = np.zeros((len(q), k), dtype=np.int64)
        sc = np.zeros((len(q), k), dtype=np.float32)
        for s in range(0, len(q), chunk):
            S = q[s : s + chunk] @ D.T
            part = np.argpartition(-S, k, axis=1)[:, :k]
            ps = np.take_along_axis(S, part, axis=1)
            o = np.argsort(-ps, axis=1, kind="stable")
            ids[s : s + chunk] = np.take_along_axis(part, o, axis=1)
            sc[s : s + chunk] = np.take_along_axis(ps, o, axis=1)
        return ids, sc
