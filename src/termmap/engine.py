"""Inference engine: maps (entity, chunk_text[, subconcept]) to ICD-10-CM and SNOMED CT."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ranker
from .corpus import SYSTEMS, description_map
from .encoders import Encoder, set_threads
from .features import QueryEncoder
from .io import load_train
from .paths import ARTIFACTS
from .pipeline import (Coupler, add_extra_features, attach_keys, bridge_variants, build_memory, make_keys,
                       shortlist, stage1_features, top_probs)
from .resources import SystemResources

KEY_OFFSET = 10_000_000  # inference keys never collide with training keys
LLM_HEADROOM_GB = 3.0  # activations + CUDA context on top of the weights


def _config(tag: str) -> dict:
    p = ARTIFACTS / f"config{tag}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _weights_gb(model_id: str) -> float | None:
    """Size of the model's safetensors in the local Hugging Face cache (None if not downloaded)."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    files = list((hub / ("models--" + model_id.replace("/", "--")) / "snapshots").glob("*/*.safetensors"))
    return sum(f.stat().st_size for f in files) / 2**30 if files else None


def _llm_fits(model_id: str) -> tuple[bool, str]:
    import torch
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    need = _weights_gb(model_id)
    free = torch.cuda.mem_get_info()[0] / 2**30
    if need is not None and free < need + LLM_HEADROOM_GB:
        return False, f"{model_id} needs ~{need + LLM_HEADROOM_GB:.0f} GB, {free:.0f} GB free on the GPU"
    return True, ""


class Engine:
    def __init__(self, dense: tuple[str, ...] = ("sapbert",), tag: str = "", verbose: bool = True,
                 llm_model: str = "", llm_shots: int = 5):
        set_threads()
        cfg = _config(tag) or {"dense": list(dense), "bridge": 1}
        dense = tuple(cfg["dense"])  # inference must use exactly the training configuration
        self.bridge = bool(cfg.get("bridge", 1))
        self.requested_tag, self.dense, self.verbose, self.llm_shots = tag, dense, verbose, llm_shots
        self.qenc = QueryEncoder({m: Encoder(m) for m in dense})
        tr = load_train()
        keys = make_keys(tr)
        tr["key"] = attach_keys(tr, keys).values
        self.train = tr
        self.res = {s: SystemResources(s, dense_models=dense) for s in SYSTEMS}
        self.mem = {s: build_memory(tr, s, self.qenc, dense) for s in SYSTEMS}
        self.coupler = Coupler(tr, self.res)
        self.desc = {s: description_map(s) for s in SYSTEMS}
        self.model_desc = {s: description_map(s, model_input=True) for s in SYSTEMS}  # what the CE / LLM saw
        self._load_model_set(tag, cfg, llm_model)

    def _load_model_set(self, tag: str, cfg: dict, llm_model: str = ""):
        """Load the tag-specific parts (rankers, calibration, cross-encoder, LLM). A tag whose LLM
        cannot run here (no GPU, not enough free memory, OOM while loading) falls back along
        cfg["fallback"] to a model set trained without it, so a shared GPU never breaks a run."""
        self.fallback_reason = ""
        while True:
            llm_id = llm_model or cfg.get("llm_model", "")
            final_stage = cfg.get("final_stage", {})
            needs_s2 = any(final_stage.get(s, 2) == 2 for s in SYSTEMS)  # stage-2 extras only if used
            self.llm, why = None, ""
            if cfg.get("llm") and llm_id and needs_s2:
                ok, why = _llm_fits(llm_id)
                if ok:
                    try:
                        from .llm_scorer import LLMScorer
                        self.llm = LLMScorer(llm_id)
                    except RuntimeError as e:  # torch.OutOfMemoryError is a RuntimeError
                        if "out of memory" not in str(e).lower():
                            raise
                        import gc

                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                        why = f"out of GPU memory while loading {llm_id}"
                if self.llm is None:
                    nxt = cfg.get("fallback")
                    if not nxt:
                        raise RuntimeError(f"model set {tag} needs {llm_id} but {why}; pick a non-LLM tag")
                    self._log(f"[engine] {tag}: {why} -> falling back to {nxt}")
                    self.fallback_reason += ("; " if self.fallback_reason else "") + f"{tag}: {why}"
                    tag, cfg, llm_model = nxt, _config(nxt), ""
                    assert tuple(cfg["dense"]) == self.dense, "fallback must share the dense configuration"
                    continue
            break
        self.tag, self.final_stage = tag, final_stage
        self.s1 = {s: ranker.load(s, "_s1" + tag) for s in SYSTEMS}
        self.s2 = {s: ranker.load(s, "_s2" + tag) for s in SYSTEMS}
        self.ce = {}
        for p in (cfg.get("extra", []) if needs_s2 else []):
            name = Path(p).name
            if name.startswith("ce_scores_train"):
                from .cross_encoder import CrossEncoder
                t = name[len("ce_scores_train"):-len(".parquet")]
                self.ce = {s: CrossEncoder(str(ARTIFACTS / f"ce_{s}{t}")) for s in SYSTEMS}
        self.calib = {}
        for s in SYSTEMS:
            cp = ARTIFACTS / f"calib_{s}{tag}.json"
            if cp.exists():
                self.calib[s] = json.loads(cp.read_text())
        self._log(f"[engine] model set {tag}" + (f" with {llm_id}" if self.llm else " (no LLM)"))

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def map(self, df: pd.DataFrame, llm_scores: pd.DataFrame | None = None) -> pd.DataFrame:
        """df needs entity, chunk_text, subconcept columns. Returns one row per input row with
        icd10cm_code, snomed_code, per-system confidence and top-5 alternatives."""
        t0 = time.time()
        df = df.copy()
        for c in ("chunk_text", "subconcept"):
            df[c] = df[c].fillna("") if c in df else ""
        if (df["subconcept"].astype(str).str.len() == 0).all():
            # input without entity types: infer them so the ranker keeps its type signal
            from . import subconcept as subc
            df["subconcept"] = subc.predict(df)
            self._log("subconcept column missing -> inferred with fallback classifier")
        keys = make_keys(df)
        keys["key"] += KEY_OFFSET
        df["key"] = attach_keys(df, keys).values
        S, F1 = {}, {}
        for s in SYSTEMS:
            mem, mem_keys = self.mem[s]
            extra = None
            if s == "ICD10CM" and self.dense and self.bridge:
                extra = bridge_variants(self.res["SNOMEDCT_US"], self.qenc, keys["entity"].tolist(), model=self.dense[0])
            F = stage1_features(self.res[s], mem, mem_keys, self.qenc, keys, loo=False,
                                extra_variants=extra).reset_index(drop=True)
            bst, cols = self.s1[s]
            F["label"] = 0
            score = bst.predict(F[cols], num_threads=ranker.PARAMS["num_threads"])
            S[s] = shortlist(F, score)
            self._log(f"[{s}] stage-1 done for {len(keys)} keys ({time.time()-t0:.0f}s)")
        if self.llm is not None and llm_scores is None:
            from .llm_scorer import neighbour_examples, score_shortlists
            ki = keys.set_index("key")
            parts = []
            for s in SYSTEMS:
                mem, mem_keys = self.mem[s]
                ex = (neighbour_examples(mem, mem_keys, self.qenc, self.dense[0], keys, self.model_desc[s], s,
                                         k=self.llm_shots, loo=False) if self.llm_shots else None)
                parts.append(score_shortlists(self.llm, s, S[s], ki, self.model_desc[s], examples=ex))
                self._log(f"[{s}] LLM shortlist scoring done ({time.time()-t0:.0f}s)")
            llm_scores = pd.concat(parts)
        extra = {s: [] for s in SYSTEMS}
        if llm_scores is not None:
            for s in SYSTEMS:
                extra[s].append(llm_scores[llm_scores.system == s][["key", "code", "llm_lp"]])
        if self.ce:
            from .cross_encoder import build_groups
            ki = keys.set_index("key")
            for s in SYSTEMS:
                extra[s].append(self.ce[s].score(build_groups(s, S[s], ki, self.model_desc[s])))
                self._log(f"[{s}] cross-encoder scoring done ({time.time()-t0:.0f}s)")
        out = {}
        for s in SYSTEMS:
            other = "SNOMEDCT_US" if s == "ICD10CM" else "ICD10CM"
            X = S[s].reset_index(drop=True)
            X = pd.concat([X, self.coupler.features(s, X, top_probs(S[other]))], axis=1)
            bst, cols = self.s2[s]
            if "llm_lp" in cols and llm_scores is None and self.final_stage.get(s, 2) == 2:
                raise ValueError("stage-2 model expects LLM scores; run on a GPU with the LLM or use a non-LLM tag")
            if self.final_stage.get(s, 2) == 2:
                X = add_extra_features(X, extra[s])
            if self.final_stage.get(s, 2) == 2:
                X["s2"] = bst.predict(X[cols], num_threads=ranker.PARAMS["num_threads"])
            else:  # CV preferred stage 1 for this system
                X["s2"] = X["s1"]
            X = X.sort_values(["key", "s2", "code"], ascending=[True, False, True], kind="stable")
            X["p2"] = np.exp(X["s2"] - X.groupby("key")["s2"].transform("max"))
            X["p2"] = X["p2"] / X.groupby("key")["p2"].transform("sum")
            g = X.groupby("key")
            top = g.head(1).set_index("key")
            second = g.nth(1).set_index("key")["s2"]
            margin = top["s2"] - second.reindex(top.index).fillna(top["s2"] - 10)
            if s in self.calib:
                conf = np.interp(margin.values, self.calib[s]["x"], self.calib[s]["y"])
            else:
                conf = top["p2"].values
            res = pd.DataFrame({
                "code": top["code"],
                "confidence": np.round(conf, 4),
                "margin": margin.round(4),
                "alternatives": g.head(5).groupby("key")["code"].apply(lambda c: "|".join(c)),
            })
            res["description"] = res["code"].map(self.desc[s])
            out[s] = res
        r = df[["key"]].copy()
        for s, pre in (("ICD10CM", "icd10cm"), ("SNOMEDCT_US", "snomed")):
            o = out[s].reindex(r["key"].values)
            r[f"{pre}_code"] = o["code"].values
            r[f"{pre}_description"] = o["description"].values
            r[f"{pre}_confidence"] = o["confidence"].values
            r[f"{pre}_margin"] = o["margin"].values
            r[f"{pre}_alternatives"] = o["alternatives"].values
        self._log(f"mapped {len(df)} rows / {len(keys)} unique terms in {time.time()-t0:.1f}s")
        return r.drop(columns="key")
