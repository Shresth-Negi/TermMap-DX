"""End-to-end staging shared by training and inference.

Stage 1  retrieval features  -> LightGBM ranker (per system)
Stage 2  stage-1 scores + cross-system coupling (+ optional LLM listwise scores)
         -> LightGBM ranker (per system) over the stage-1 shortlist

Queries are keyed by (entity, chunk_text, subconcept). Each key is mapped in
*both* systems so the two predictions can inform each other.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .features import Queries, QueryEncoder, TrainMemory, build_features, clean_entity, context_without_mention
from .paths import ARTIFACTS
from .resources import SystemResources

DX_SUBCONCEPTS = {"DiagnosisCode", "", None}
SHORTLIST = 20


def make_keys(df: pd.DataFrame) -> pd.DataFrame:
    k = df[["entity", "chunk_text", "subconcept"]].drop_duplicates().reset_index(drop=True)
    k["key"] = np.arange(len(k))
    return k


def attach_keys(df: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    m = keys.set_index(["entity", "chunk_text", "subconcept"])["key"]
    return pd.Series(m.reindex(pd.MultiIndex.from_frame(df[["entity", "chunk_text", "subconcept"]])).values,
                     index=df.index)


def system_keys(keys: pd.DataFrame, system: str) -> pd.DataFrame:
    if system == "ICD10CM":
        return keys[keys["subconcept"].isin(DX_SUBCONCEPTS)].reset_index(drop=True)
    return keys.reset_index(drop=True)


def _encode_cached(qenc: QueryEncoder, model: str, texts: list[str]) -> np.ndarray:
    """Query-side embeddings of the training memory, cached on disk per model, device and text set
    (avoids re-embedding ~10k terms at every cold start; CPU and GPU vectors are never mixed)."""
    enc = qenc.encoders[model]
    h = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:16]
    p = ARTIFACTS / f"memvec_{model}_{enc.device}_{enc.precision}_{h}.npy"
    if p.exists():
        return np.load(p)
    # straight through the encoder, not the shared query cache: a query that equals a training
    # term must be embedded the same way whether or not this cache file already existed
    v = enc.encode(texts, batch_size=256, query=True)
    np.save(p, v)
    return v


def build_memory(train: pd.DataFrame, system: str, qenc: QueryEncoder, dense: tuple[str, ...]) -> tuple[TrainMemory, np.ndarray]:
    t = train[train["coding_system"] == system].reset_index(drop=True)
    texts = [clean_entity(e)[0] for e in t["entity"]]
    ctxs = [context_without_mention(e, c) for e, c in zip(t["entity"], t["chunk_text"])]
    mem = TrainMemory(texts, t["code"].to_numpy(dtype=object), {m: _encode_cached(qenc, m, texts) for m in dense},
                      {m: _encode_cached(qenc, m, ctxs) for m in dense[:1]})
    return mem, t["key"].values if "key" in t else np.full(len(t), -1)


def bridge_variants(res_sn: SystemResources, qenc: QueryEncoder, entities: list[str], model: str = "sapbert",
                    k: int = 3, min_sim: float = 0.75) -> list[list[str]]:
    """Use SNOMED as a synonym thesaurus: preferred terms of the top-k SNOMED concepts
    (dense match on the cleaned entity) become extra ICD query variants."""
    if model not in res_sn.dense or model not in qenc.encoders:
        return [[] for _ in entities]
    texts = [clean_entity(e)[0] for e in entities]
    ids, sc = res_sn.dense_search(model, qenc(model, texts), k=10)
    out = []
    for row, srow in zip(ids, sc):
        seen = []
        for d, s in zip(row, srow):
            if s < min_sim:
                break
            for c in res_sn.desc_codes[d]:
                p = res_sn.texts[res_sn.pref_desc.get(c, d)]
                if p not in seen:
                    seen.append(p)
            if len(seen) >= k:
                break
        out.append(seen[:k])
    return out


def stage1_features(res: SystemResources, mem: TrainMemory, mem_keys: np.ndarray, qenc: QueryEncoder,
                    qkeys: pd.DataFrame, loo: bool, extra_variants: list[list[str]] | None = None) -> pd.DataFrame:
    """Features for every key in qkeys. With loo=True, a key's own memory row is excluded."""
    excl = None
    if loo:
        pos = {k: i for i, k in enumerate(mem_keys)}
        excl = np.array([pos.get(k, -1) for k in qkeys["key"]], dtype=np.int64)
    Q = Queries(qkeys["entity"].tolist(), qkeys["chunk_text"].tolist(), qkeys["subconcept"].tolist())
    F = build_features(res, mem, qenc, Q, exclude_self=excl, extra_variants=extra_variants)
    F["key"] = qkeys["key"].values[F["qid"].values]
    return F


# ---------------------------------------------------------------- coupling
class Coupler:
    """Cross-system signals: co-occurrence of gold pairs in training and
    dense similarity of preferred descriptions across the two dictionaries."""

    def __init__(self, train: pd.DataFrame, res: dict[str, SystemResources], dense: str = "sapbert"):
        self.res, self.dense = res, dense
        piv = train.pivot_table(index="key", columns="coding_system", values="code", aggfunc="first").dropna()
        self.pairs = piv  # key -> ICD10CM, SNOMEDCT_US
        self.cnt = Counter(zip(piv["SNOMEDCT_US"], piv["ICD10CM"]))
        self.marg = {"SNOMEDCT_US": Counter(piv["SNOMEDCT_US"]), "ICD10CM": Counter(piv["ICD10CM"])}
        self.pair_key = defaultdict(set)
        for k, s, i in zip(piv.index, piv["SNOMEDCT_US"], piv["ICD10CM"]):
            self.pair_key[(s, i)].add(k)

    def _vec(self, system: str, code: str):
        r = self.res[system]
        d = r.pref_desc.get(code)
        if d is None or self.dense not in r.dense:
            return None
        return r.dense[self.dense][d]

    def cond(self, system: str, cand: str, other: str, key: int) -> float:
        """P(cand | other) with the query's own pair removed (leave-one-out)."""
        s, i = (other, cand) if system == "ICD10CM" else (cand, other)
        c = self.cnt.get((s, i), 0) - (1 if key in self.pair_key.get((s, i), ()) else 0)
        own_other = self.pairs["SNOMEDCT_US" if system == "ICD10CM" else "ICD10CM"].get(key)
        m = self.marg["SNOMEDCT_US" if system == "ICD10CM" else "ICD10CM"].get(other, 0) - (1 if own_other == other else 0)
        return c / m if m > 0 else 0.0

    def features(self, system: str, F: pd.DataFrame, other_top: pd.DataFrame) -> pd.DataFrame:
        """F: stage-2 rows (key, code). other_top: (key, code, p) top-k of the other system."""
        other_sys = "SNOMEDCT_US" if system == "ICD10CM" else "ICD10CM"
        ot = other_top.groupby("key")
        groups = {k: (g["code"].tolist(), g["p"].values) for k, g in ot}
        cooc, cooc_top1, sim_w, sim_top1 = [], [], [], []
        vcache: dict = {}

        def vec(sys_, c):
            k = (sys_, c)
            if k not in vcache:
                vcache[k] = self._vec(sys_, c)
            return vcache[k]

        for key, code in zip(F["key"].values, F["code"].values):
            g = groups.get(key)
            if g is None:
                cooc.append(np.nan); cooc_top1.append(np.nan); sim_w.append(np.nan); sim_top1.append(np.nan)
                continue
            ocodes, ps = g
            cs = [self.cond(system, code, o, key) for o in ocodes]
            cooc.append(float(np.dot(cs, ps))); cooc_top1.append(cs[0])
            v = vec(system, code)
            sims = [float(v @ vec(other_sys, o)) if v is not None and vec(other_sys, o) is not None else 0.0
                    for o in ocodes]
            sim_w.append(float(np.dot(sims, ps))); sim_top1.append(sims[0])
        out = pd.DataFrame({"xs_cooc": cooc, "xs_cooc_top1": cooc_top1, "xs_sim": sim_w, "xs_sim_top1": sim_top1},
                           index=F.index)
        return out


def shortlist(F: pd.DataFrame, score: np.ndarray, n: int = SHORTLIST) -> pd.DataFrame:
    d = F.copy()
    d["s1"] = score
    d = d.sort_values(["qid", "s1", "code"], ascending=[True, False, True], kind="stable")
    d["s1_rank"] = d.groupby("qid").cumcount()
    d = d[d["s1_rank"] < n].copy()
    mx = d.groupby("qid")["s1"].transform("max")
    e = np.exp(d["s1"] - mx)
    d["s1_p"] = e / e.groupby(d["qid"]).transform("sum")
    d["s1_gap"] = mx - d["s1"]
    return d


def top_probs(S: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return S[S["s1_rank"] < k][["key", "code", "s1_p"]].rename(columns={"s1_p": "p"})


def add_extra_features(X: pd.DataFrame, extras: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge shortlist-level score tables (key, code, <features>) and derive per-key rank / gap."""
    for E in extras:
        cols = [c for c in E.columns if c not in ("system", "key", "code")]
        X = X.merge(E[["key", "code"] + cols], on=["key", "code"], how="left")
        for c in cols:
            X[f"{c}_rank"] = X.groupby("key")[c].rank(ascending=False, method="min")
            X[f"{c}_gap"] = X.groupby("key")[c].transform("max") - X[c]
    return X
