"""Fine-tuned cross-encoder over the stage-1 shortlist (listwise softmax loss).

Query side: entity | entity type | local context.  Candidate side: preferred
description (ICD code prefixed). Initialised from SapBERT. Trained with the
gold candidate as the positive and the other shortlist entries as hard
negatives. Out-of-fold scores (same folds as the rankers) are used as a stage-2
feature; a model trained on all folds scores new data at inference.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .features import clean_entity, context_window

BASE = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
TOPN = 16
MAXLEN = 96


def query_text(entity: str, chunk: str, subconcept: str) -> str:
    return f"{clean_entity(entity)[0]} | {subconcept or 'unknown'} | {context_window(entity, chunk, 60)}"


def cand_text(system: str, code: str, desc: str) -> str:
    return f"{code} {desc}" if system == "ICD10CM" else desc


def build_groups(system: str, shortlist: pd.DataFrame, keyinfo: pd.DataFrame, desc_of: dict,
                 topn: int = TOPN):
    """-> list of (key, query_text, [cand texts], [codes], gold_index or -1)."""
    sl = shortlist[shortlist["s1_rank"] < topn].sort_values(["key", "s1_rank"])
    groups = []
    for key, g in sl.groupby("key", sort=True):
        k = keyinfo.loc[key]
        q = query_text(k["entity"], k["chunk_text"], k["subconcept"])
        codes = g["code"].tolist()
        lab = g["label"].values if "label" in g else np.zeros(len(g))
        gi = int(np.argmax(lab)) if lab.max() > 0 else -1
        groups.append((key, q, [cand_text(system, c, desc_of.get(c, c)) for c in codes], codes, gi))
    return groups


class CrossEncoder:
    def __init__(self, path: str = BASE, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(self.device)

    def _enc(self, qs, cs):
        return self.tok(qs, cs, padding=True, truncation="longest_first", max_length=MAXLEN,
                        return_tensors="pt").to(self.device)

    def fit(self, groups, epochs: int = 2, lr: float = 3e-5, q_per_batch: int = 16, seed: int = 13,
            log_every: int = 200):
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        train = [g for g in groups if g[4] >= 0]
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(train) / q_per_batch)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
        self.model.train()
        rng = random.Random(seed)
        step, t0 = 0, time.time()
        for ep in range(epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            for b in range(0, len(order), q_per_batch):
                batch = [train[i] for i in order[b : b + q_per_batch]]
                qs, cs, sizes, golds = [], [], [], []
                for _, q, cands, _, gi in batch:
                    qs += [q] * len(cands); cs += cands; sizes.append(len(cands)); golds.append(gi)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    logits = self.model(**self._enc(qs, cs)).logits.squeeze(-1).float()
                loss, off = 0.0, 0
                for n, gi in zip(sizes, golds):
                    loss = loss + torch.nn.functional.cross_entropy(logits[off : off + n].unsqueeze(0),
                                                                    torch.tensor([gi], device=self.device))
                    off += n
                loss = loss / len(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    print(f"    ce step {step}/{steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        self.model.eval()
        return self

    @torch.inference_mode()
    def score(self, groups, pairs_per_batch: int = 512) -> pd.DataFrame:
        self.model.eval()
        flat = [(key, q, c, code) for key, q, cands, codes, _ in groups for c, code in zip(cands, codes)]
        out = np.zeros(len(flat), dtype=np.float32)
        order = np.argsort([len(f[1]) + len(f[2]) for f in flat])
        for s in range(0, len(flat), pairs_per_batch):
            idx = order[s : s + pairs_per_batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                lg = self.model(**self._enc([flat[i][1] for i in idx], [flat[i][2] for i in idx])).logits
            out[idx] = lg.squeeze(-1).float().cpu().numpy()
        df = pd.DataFrame({"key": [f[0] for f in flat], "code": [f[3] for f in flat], "ce": out})
        df["ce_lp"] = df["ce"] - df.groupby("key")["ce"].transform(lambda v: np.log(np.exp(v - v.max()).sum()) + v.max())
        return df

    def save(self, path):
        self.model.save_pretrained(path); self.tok.save_pretrained(path)
