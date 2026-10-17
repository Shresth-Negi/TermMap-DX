"""Candidate generation + feature extraction for the learning-to-rank stage.

For one coding system and a batch of queries (entity, context, subconcept) this
module
  1. cleans the entity and builds abbreviation-expanded variants,
  2. pulls candidate codes from several independent retrievers
     (exact key, char TF-IDF, word TF-IDF, dense encoders, training neighbours,
     codes literally written in the text),
  3. scores *every* candidate under *every* signal (not only the retriever that
     proposed it), and
  4. emits one feature row per (query, candidate code).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .abbrev import expand
from .lexical import LexicalIndex
from .normalize import basic_norm, key_norm
from .resources import SystemResources

STOP = set("""a an the of and or in on at to for with without by from as is was are be been has have had
not no nos this that these those due other unspecified specified type stage status s/p history""".split())
LAT_WORDS = {"left": 1, "lt": 1, "right": 2, "rt": 2, "bilateral": 3, "bilat": 3, "b/l": 3, "both": 3, "bil": 3}
SUBCONCEPTS = ["DiagnosisCode", "ProcedureCode", "LabService", "EncounterType", "Device", "MedicationProfileMedication"]
SEMTAGS = ["disorder", "finding", "procedure", "observable entity", "physical object", "regime/therapy",
           "qualifier value", "situation", "substance", "record artifact", "morphologic abnormality",
           "body structure", "environment", "assessment scale", "event", "attribute", "occupation", "person",
           "organism"]

_ICD_IN_TEXT = re.compile(r"\b([a-tv-z][0-9][0-9a-z](?:\.[0-9a-z]{1,4})?)\b")
_HCC = re.compile(r"\((?:cms[/\- ]?)?hcc\)|\bcms[/\- ]hcc\b")
_LEAD = re.compile(r"^[\s\-\*•\d\)\.:]+(?=[a-z])")


def clean_entity(raw: str) -> tuple[str, list[str]]:
    s = basic_norm(raw)
    s = _HCC.sub(" ", s)
    mentioned = [m.upper() for m in _ICD_IN_TEXT.findall(s)]
    s = _LEAD.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")
    return s or basic_norm(raw), mentioned


def lat_of(text: str) -> int:
    v = 0
    for t in re.findall(r"[a-z/]+", text):
        if t in LAT_WORDS:
            v = 3 if (v and v != LAT_WORDS[t]) else LAT_WORDS[t]
    return v


_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}
_NUM = re.compile(r"\d+")


def numbers(text: str) -> set[str]:
    t = key_norm(text).split()
    out = set(_NUM.findall(" ".join(t)))
    out |= {_ROMAN[x] for x in t if x in _ROMAN and x != "i" and x != "v"}
    return {n.lstrip("0") or "0" for n in out}


def content_tokens(text: str) -> set[str]:
    return {t for t in key_norm(text).split() if t not in STOP and len(t) > 1}


_NEG_CUE = re.compile(r"\b(no|not|denies|denied|deny|negative for|neg for|without|absent|free of|resolved|"
                      r"never|none|r/o|rule out|ruled out|unlikely)\b[^.;:]{0,40}$")
_NEG_CAND = re.compile(r"\b(absent|absence|no|not|negative|denies|without|never|free of|normal)\b")
_HIST = re.compile(r"\b(history of|h/o|hx of|past|previous|prior|status post|s/p|resolved)\b[^.;:]{0,30}$")
_HIST_IN = re.compile(r"\b(history of|h/o|hx of|hx|past|previous|prior|status post|s/p|resolved|old)\b")
_FAM = re.compile(r"\b(family history|fh|fhx|mother|father|sister|brother|mom|dad)\b[^.;]{0,40}$")


def context_flags(entity: str, chunk: str) -> tuple[int, int, int]:
    """(negated, historical, family) cues in the text immediately before the entity."""
    c = basic_norm(chunk)
    e = basic_norm(entity)
    i = c.find(e) if e else -1
    pre = c[max(0, i - 60) : i] if i >= 0 else ""
    return (int(bool(_NEG_CUE.search(pre)) or e.startswith(("no ", "not ", "denies ", "negative "))),
            int(bool(_HIST.search(pre)) or bool(_HIST_IN.search(e))), int(bool(_FAM.search(pre))))


def context_without_mention(entity: str, chunk: str, width: int = 80) -> str:
    w = context_window(entity, chunk, width)
    return re.sub(r"\s+", " ", w.replace(basic_norm(entity), " ")).strip() or "none"


def context_window(entity: str, chunk: str, width: int = 80) -> str:
    c = basic_norm(chunk)
    e = basic_norm(entity)
    i = c.find(e) if e else -1
    if i < 0:
        return c[: 2 * width]
    return c[max(0, i - width) : i + len(e) + width]


@dataclass
class TrainMemory:
    """Labelled training examples of one system, used for neighbour transfer and priors."""

    texts: list[str]
    codes: np.ndarray
    dense: dict[str, np.ndarray] = field(default_factory=dict)
    ctx_dense: dict[str, np.ndarray] = field(default_factory=dict)  # context (mention removed) embeddings

    def __post_init__(self):
        self.char = LexicalIndex("char", "mem").build(self.texts)
        self.prior = Counter(self.codes.tolist())
        self.block_prior = Counter(c[:3] for c in self.codes.tolist())


class QueryEncoder:
    """Wraps dense encoders with an in-memory text->vector cache."""

    def __init__(self, encoders: dict):
        self.encoders = encoders
        self.cache: dict[str, dict[str, np.ndarray]] = {m: {} for m in encoders}

    def __call__(self, model: str, texts: list[str]) -> np.ndarray:
        cache = self.cache[model]
        todo = sorted({t for t in texts if t not in cache})
        if todo:
            vecs = self.encoders[model].encode(todo, batch_size=256, query=True)
            for t, v in zip(todo, vecs):
                cache[t] = v
        dim = next(iter(cache.values())).shape[0]
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, dim), np.float32)


@dataclass
class Queries:
    entity: list[str]
    context: list[str]
    subconcept: list[str | None]


K_CHAR, K_WORD, K_DENSE, K_VAR, K_NN = 50, 20, 60, 25, 20


def build_features(res: SystemResources, mem: TrainMemory, qenc: QueryEncoder, Q: Queries,
                   exclude_self: np.ndarray | None = None,
                   extra_variants: list[list[str]] | None = None) -> pd.DataFrame:
    """One row per (query, candidate). ``exclude_self[i]`` = index into mem to skip (LOO)."""
    n = len(Q.entity)
    sys_ = res.system
    cleaned = [clean_entity(e) for e in Q.entity]
    v0 = [c for c, _ in cleaned]
    mentioned = [m for _, m in cleaned]
    variants = [expand(t) for t in v0]
    if extra_variants is not None:
        variants = [list(dict.fromkeys(v + [x for x in ev if x != t])) for v, ev, t in zip(variants, extra_variants, v0)]
    ctxw = [context_window(e, c) for e, c in zip(Q.entity, Q.context)]
    qflags = [context_flags(e, c) for e, c in zip(Q.entity, Q.context)]
    dense_models = [m for m in res.dense if m in qenc.encoders]

    # ---- query representations ----
    flat_var = [v for vs in variants for v in vs]
    var_owner = np.array([i for i, vs in enumerate(variants) for _ in vs], dtype=np.int64)
    qd = {m: qenc(m, v0) for m in dense_models}
    qd_var = {m: qenc(m, flat_var) for m in dense_models}
    qd_ctx = {m: qenc(m, ctxw) for m in dense_models[:1]}
    # context with the mention itself removed: used to pick between readings of an abbreviation
    ctx_noent = [context_without_mention(e, c) for e, c in zip(Q.entity, Q.context)]
    qd_ctxn = {m: qenc(m, ctx_noent) for m in dense_models[:1]}
    qc = res.char.transform(v0)
    qc_var = res.char.transform(flat_var) if flat_var else None
    qw = res.word.transform(v0)

    # ---- candidate generation ----
    cand: list[dict[str, int]] = [defaultdict(int) for _ in range(n)]
    SRC = {"char": 1, "word": 2, "dense": 4, "nn": 8, "exact": 16, "mention": 32, "var": 64}

    def add_desc_hits(ids, flag, owner=None):
        for r, row in enumerate(ids):
            i = owner[r] if owner is not None else r
            for d in row:
                if d < 0:
                    continue
                for c in res.desc_codes[d]:
                    cand[i][c] |= flag

    ids, _ = res.char.search(v0, k=K_CHAR)
    add_desc_hits(ids, SRC["char"])
    ids, _ = res.word.search(v0, k=K_WORD)
    add_desc_hits(ids, SRC["word"])
    if flat_var:
        ids, _ = res.char.search(flat_var, k=K_VAR)
        add_desc_hits(ids, SRC["var"], var_owner)
    for m in dense_models:
        ids, _ = res.dense_search(m, qd[m], k=K_DENSE)
        add_desc_hits(ids, SRC["dense"])
        if flat_var:
            ids, _ = res.dense_search(m, qd_var[m], k=K_VAR)
            add_desc_hits(ids, SRC["var"], var_owner)
    for i in range(n):
        for t in [v0[i]] + variants[i]:
            for c in res.key2codes.get(key_norm(t), ()):
                cand[i][c] |= SRC["exact"]
        for c in mentioned[i]:
            if c in res.code_set:
                cand[i][c] |= SRC["mention"]

    # training neighbours (char + dense), neighbour gold codes become candidates
    nn_feats = [defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0]) for _ in range(n)]  # cmax,csum,dmax,dsum,cnt
    nn_ctx = [defaultdict(lambda: [0.0, 0.0]) for _ in range(n)]  # max(ent*ctx sim), max ctx sim
    kk = K_NN + 1
    nid, nsc = mem.char.search(v0, k=kk)
    for i in range(n):
        taken = 0
        for j, s in zip(nid[i], nsc[i]):
            if j < 0 or (exclude_self is not None and j == exclude_self[i]):
                continue
            c = mem.codes[j]
            f = nn_feats[i][c]
            f[0] = max(f[0], float(s)); f[1] += float(s); f[4] += 1
            cand[i][c] |= SRC["nn"]
            taken += 1
            if taken >= K_NN:
                break
    if dense_models and dense_models[0] in mem.dense:
        m = dense_models[0]
        M = mem.dense[m]
        for s0 in range(0, n, 2048):
            S = qd[m][s0 : s0 + 2048] @ M.T
            if exclude_self is not None:
                ex = exclude_self[s0 : s0 + 2048]
                ok = ex >= 0
                S[np.arange(len(S))[ok], ex[ok]] = -1
            top = np.argsort(-S, axis=1)[:, :K_NN]
            Cq = qd_ctxn[m][s0 : s0 + 2048] if m in mem.ctx_dense else None
            for r in range(len(S)):
                i = s0 + r
                csim = mem.ctx_dense[m][top[r]] @ Cq[r] if Cq is not None else None
                for t, j in enumerate(top[r]):
                    c = mem.codes[j]
                    s = float(S[r, j])
                    f = nn_feats[i][c]
                    f[2] = max(f[2], s); f[3] += max(s, 0.0) ** 4
                    cand[i][c] |= SRC["nn"]
                    if csim is not None:
                        g = nn_ctx[i][c]
                        g[0] = max(g[0], s * float(csim[t])); g[1] = max(g[1], float(csim[t]))

    # ---- per-candidate scoring ----
    rows = []
    for i in range(n):
        codes = [c for c in cand[i] if c in res.code_descs]
        if not codes:
            continue
        descs = [res.code_descs[c] for c in codes]
        lens = np.array([len(d) for d in descs])
        flat = np.concatenate(descs)
        owner = np.repeat(np.arange(len(codes)), lens)
        bounds = np.concatenate([[0], np.cumsum(lens)])

        def seg_max(v):
            return np.maximum.reduceat(v, bounds[:-1]) if len(v) else v

        def seg_argmax(v):
            out = np.empty(len(codes), dtype=np.int64)
            for k in range(len(codes)):
                seg = v[bounds[k] : bounds[k + 1]]
                out[k] = flat[bounds[k] + int(np.argmax(seg))]
            return out

        feats: dict[str, np.ndarray] = {}
        cs = np.asarray(res.char.mat[flat].dot(qc[i].T).todense()).ravel()
        feats["char_max"] = seg_max(cs)
        ws = np.asarray(res.word.mat[flat].dot(qw[i].T).todense()).ravel()
        feats["word_max"] = seg_max(ws)
        vi = np.where(var_owner == i)[0]
        if len(vi):
            cv = np.asarray(res.char.mat[flat].dot(qc_var[vi].T).todense()).max(axis=1)
            feats["char_var"] = seg_max(np.asarray(cv).ravel())
        else:
            feats["char_var"] = feats["char_max"]
        best_desc = None
        for m in dense_models:
            E = res.dense[m][flat]
            ds = E @ qd[m][i]
            feats[f"{m}_max"] = seg_max(ds)
            pref = np.array([res.pref_desc.get(c, -1) for c in codes])
            feats[f"{m}_pref"] = res.dense[m][pref] @ qd[m][i]
            if len(vi):
                dvm = E @ qd_var[m][vi].T
                feats[f"{m}_var"] = seg_max(dvm.max(axis=1))
            else:
                feats[f"{m}_var"] = feats[f"{m}_max"]
            if best_desc is None and m in qd_ctxn:
                # which reading (raw string or an expansion) explains each candidate, and how well
                # does that reading fit the surrounding context?
                c_raw = float(qd[m][i] @ qd_ctxn[m][i])
                if len(vi):
                    per_var = np.maximum.reduceat(dvm, bounds[:-1], axis=0)  # codes x variants
                    vsim = qd_var[m][vi] @ qd_ctxn[m][i]
                    use_var = per_var.max(axis=1) > feats[f"{m}_max"]
                    rc = np.where(use_var, vsim[per_var.argmax(axis=1)], c_raw)
                    best_r = max(c_raw, float(vsim.max()))
                else:
                    rc = np.full(len(codes), c_raw)
                    best_r = c_raw
                feats["reading_ctx"] = rc
                feats["reading_ctx_gap"] = best_r - rc
            if best_desc is None:
                best_desc = seg_argmax(ds)
                feats["ctx_dense"] = res.dense[m][best_desc] @ qd_ctx[m][i]
        if best_desc is None:
            best_desc = seg_argmax(cs)

        q = v0[i]
        qtok = content_tokens(q)
        qlat = lat_of(q)
        ctx_tok = content_tokens(ctxw[i]) - qtok
        qnum = numbers(q)
        f_rat, f_tset, f_part, f_tsort, qcov, dcov, dextra, ctxov, clat = ([] for _ in range(9))
        nmatch, nconf = [], []
        for d in best_desc:
            t = res.texts[d]
            f_rat.append(fuzz.ratio(q, t)); f_tset.append(fuzz.token_set_ratio(q, t))
            f_part.append(fuzz.partial_ratio(q, t)); f_tsort.append(fuzz.token_sort_ratio(q, t))
            dt = content_tokens(t)
            inter = len(qtok & dt)
            qcov.append(inter / max(1, len(qtok))); dcov.append(inter / max(1, len(dt)))
            dextra.append(len(dt - qtok)); ctxov.append(len((dt - qtok) & ctx_tok))
            clat.append(lat_of(t))
            dnum = numbers(t)
            nmatch.append(len(qnum & dnum)); nconf.append(int(bool(qnum) and bool(dnum) and not (qnum & dnum)))
        clat = np.array(clat)
        feats.update(num_match=np.array(nmatch), num_conflict=np.array(nconf, dtype=np.int8))
        cneg = np.array([bool(_NEG_CAND.search(res.texts[d])) for d in best_desc], dtype=np.int8)
        chist = np.array([("history" in res.texts[d]) for d in best_desc], dtype=np.int8)
        feats.update(cand_neg=cneg, cand_hist=chist, neg_match=(cneg == qflags[i][0]).astype(np.int8),
                     hist_match=(chist == qflags[i][1]).astype(np.int8))
        feats.update(fz_ratio=np.array(f_rat), fz_tset=np.array(f_tset), fz_partial=np.array(f_part),
                     fz_tsort=np.array(f_tsort), tok_qcov=np.array(qcov), tok_dcov=np.array(dcov),
                     tok_dextra=np.array(dextra), ctx_overlap=np.array(ctxov), cand_lat=clat,
                     lat_match=((clat == qlat) & (qlat > 0)).astype(np.int8),
                     lat_conflict=((clat > 0) & (qlat > 0) & (clat != qlat)).astype(np.int8),
                     lat_missing=((clat > 0) & (qlat == 0)).astype(np.int8))

        nf = np.array([nn_feats[i][c] if c in nn_feats[i] else [0, 0, 0, 0, 0] for c in codes], dtype=np.float32)
        if sys_ == "ICD10CM":
            # hierarchy: neighbour votes for the candidate's 3-character category
            cat = defaultdict(lambda: np.zeros(5, dtype=np.float32))
            for c, f in nn_feats[i].items():
                a = cat[c[:3]]
                a[0] = max(a[0], f[0]); a[1] += f[1]; a[2] = max(a[2], f[2]); a[3] += f[3]; a[4] += f[4]
            cf = np.array([cat[c[:3]] if c[:3] in cat else np.zeros(5) for c in codes], dtype=np.float32)
            feats.update(nn_cat_char_max=cf[:, 0], nn_cat_char_sum=cf[:, 1], nn_cat_dense_max=cf[:, 2],
                         nn_cat_dense_sum=cf[:, 3], nn_cat_cnt=cf[:, 4])
        feats.update(nn_char_max=nf[:, 0], nn_char_sum=nf[:, 1], nn_dense_max=nf[:, 2],
                     nn_dense_sum=nf[:, 3], nn_cnt=nf[:, 4])
        if mem.ctx_dense:
            gc = np.array([nn_ctx[i][c] if c in nn_ctx[i] else [0.0, 0.0] for c in codes], dtype=np.float32)
            feats.update(nn_entctx_max=gc[:, 0], nn_ctx_max=gc[:, 1])
        self_code = mem.codes[exclude_self[i]] if exclude_self is not None and exclude_self[i] >= 0 else None
        feats["prior"] = np.log1p([mem.prior.get(c, 0) - (c == self_code) for c in codes])
        feats["src"] = np.array([cand[i][c] for c in codes])
        feats["exact"] = ((feats["src"] & SRC["exact"]) > 0).astype(np.int8)
        feats["mention"] = ((feats["src"] & SRC["mention"]) > 0).astype(np.int8)
        feats["n_desc"] = lens

        if sys_ == "ICD10CM":
            pref_txt = [res.texts[res.pref_desc[c]] for c in codes]
            feats["billable"] = np.array([int(bool(res.billable.get(c, False))) for c in codes])
            feats["code_len"] = np.array([len(c.replace(".", "")) for c in codes])
            feats["chapter"] = np.array([ord(c[0]) - 65 for c in codes])
            sb = self_code[:3] if self_code else None
            feats["block_prior"] = np.log1p([mem.block_prior.get(c[:3], 0) - (c[:3] == sb) for c in codes])
            feats["is_unspec"] = np.array([("unspecified" in t) or res.any_unspec.get(c, False)
                                           for t, c in zip(pref_txt, codes)], dtype=np.int8)
            feats["is_other"] = np.array([t.startswith("other") or " other " in t or res.any_other.get(c, False)
                                          for t, c in zip(pref_txt, codes)], dtype=np.int8)
            feats["enc7"] = np.array([{"A": 1, "D": 2, "S": 3}.get(c[-1], 0) if len(c.replace(".", "")) == 7 else 0
                                      for c in codes])
        else:
            feats["semtag"] = np.array([SEMTAGS.index(res.semtag.get(c)) if res.semtag.get(c) in SEMTAGS
                                        else len(SEMTAGS) for c in codes])
            feats["best_order"] = np.array([res.desc_order.get((c, d), 9) for c, d in zip(codes, best_desc)])
            feats["best_is_pref"] = np.array([int(res.pref_desc.get(c) == d) for c, d in zip(codes, best_desc)])

        df = pd.DataFrame(feats)
        df.insert(0, "code", codes)
        df.insert(0, "qid", i)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    # query-level features and within-query ranks / gaps
    g = out.groupby("qid")
    for col in [c for c in out.columns if c.endswith("_max") or c.endswith("_var") or c in ("fz_tset", "prior")]:
        out[f"{col}_rank"] = g[col].rank(ascending=False, method="min")
        out[f"{col}_gap"] = g[col].transform("max") - out[col]
    qlen = np.array([len(t) for t in v0]); qntok = np.array([len(t.split()) for t in v0])
    out["q_len"] = qlen[out.qid]; out["q_ntok"] = qntok[out.qid]
    out["q_digit"] = np.array([any(ch.isdigit() for ch in t) for t in v0])[out.qid].astype(np.int8)
    out["q_abbr"] = np.array([(len(t) <= 5 and " " not in t) for t in v0])[out.qid].astype(np.int8)
    out["q_nvar"] = np.array([len(v) for v in variants])[out.qid]
    out["q_lat"] = np.array([lat_of(t) for t in v0])[out.qid]
    qf = np.array(qflags)
    out["q_neg"] = qf[out.qid, 0]; out["q_hist"] = qf[out.qid, 1]; out["q_fam"] = qf[out.qid, 2]
    sc = np.array([SUBCONCEPTS.index(s) if s in SUBCONCEPTS else -1 for s in Q.subconcept])
    out["subconcept"] = sc[out.qid]
    out["n_cand"] = g["code"].transform("size")
    return out
