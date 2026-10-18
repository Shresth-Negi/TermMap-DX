"""Listwise LLM scoring of the ranker's shortlist (local open-weight model).

For each query the top-N candidates are shown as lettered options; one forward
pass reads the next-token distribution over the option letters. The resulting
probability per candidate is fed back to the ranker as a feature. No sampling,
no generation: fully deterministic for fixed weights.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDEFGHIJKLMNOP"

SYSTEM_PROMPT = (
    "You are an expert medical coder. Map the clinical term extracted from a chart to the single best "
    "{system} concept. Use the surrounding text to resolve abbreviations, negation and laterality. "
    "Prefer the concept a professional coder would assign: do not add specificity the text does not "
    "support, but use specificity the text states. Answer with the option letter only."
)
SYSTEM_NAMES = {"ICD10CM": "ICD-10-CM", "SNOMEDCT_US": "SNOMED CT"}
KIND_HINT = {
    "DiagnosisCode": "a diagnosis, condition, sign or symptom",
    "ProcedureCode": "a procedure, therapy or intervention (or related finding/observable)",
    "LabService": "a laboratory test or measurement",
    "EncounterType": "an encounter, social history or context item",
    "Device": "a device or physical object",
}


def build_prompt(tok, system: str, entity: str, context: str, subconcept: str, options: list[str],
                 examples: list[tuple[str, str]] | None = None) -> str:
    ctx = context.replace("\n", " ")
    if len(ctx) > 600:
        i = ctx.lower().find(entity.lower())
        s = max(0, i - 300) if i >= 0 else 0
        ctx = ctx[s : s + 600]
    kind = KIND_HINT.get(subconcept or "", "a clinical concept")
    opts = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(options))
    ex = ""
    if examples:
        ex = ("How our coders mapped similar terms (reference only; the answer may differ):\n"
              + "\n".join(f"- \"{t}\" -> {d}" for t, d in examples) + "\n\n")
    user = (f"{ex}Clinical term: \"{entity}\"\nEntity type: {kind}\nContext: \"{ctx}\"\n\n"
            f"Candidate {SYSTEM_NAMES[system]} concepts:\n{opts}\n\nWhich option is the correct code for the term?")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(system=SYSTEM_NAMES[system])},
            {"role": "user", "content": user}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + "Answer: "


class LLMScorer:
    def __init__(self, model_id: str, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if device == "cuda":  # load straight onto the GPU (no CPU copy of the weights)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        self.device = device
        self.letter_ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    @torch.inference_mode()
    def score(self, prompts: list[str], n_opts: list[int], batch_size: int = 16) -> list[np.ndarray]:
        order = np.argsort([-len(p) for p in prompts])
        out: list[np.ndarray | None] = [None] * len(prompts)
        for s in range(0, len(prompts), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            sel = logits[:, self.letter_ids]
            for r, i in enumerate(idx):
                v = sel[r, : n_opts[i]]
                out[i] = torch.log_softmax(v, dim=-1).cpu().numpy()
        return out


LLM_TOPN = 10


def option_text(system: str, code: str, desc: str) -> str:
    return f"{code} {desc}" if system == "ICD10CM" else desc


def score_shortlists(scorer: "LLMScorer", system: str, shortlist, keyinfo, desc_of: dict,
                     topn: int = LLM_TOPN, batch_size: int = 16, examples: dict | None = None):
    """shortlist: rows (key, code, s1_rank); keyinfo: key-indexed entity/chunk_text/subconcept.
    Returns DataFrame(system, key, code, llm_lp)."""
    import pandas as pd

    sl = shortlist[shortlist["s1_rank"] < topn].sort_values(["key", "s1_rank"])
    prompts, nopt, meta = [], [], []
    for key, g in sl.groupby("key", sort=True):
        codes = g["code"].tolist()
        k = keyinfo.loc[key]
        opts = [option_text(system, c, desc_of.get(c, c)) for c in codes]
        ex = examples.get(key) if examples else None
        prompts.append(build_prompt(scorer.tok, system, k["entity"], k["chunk_text"], k["subconcept"], opts, ex))
        nopt.append(len(codes))
        meta.append((key, codes))
    lps = scorer.score(prompts, nopt, batch_size=batch_size)
    rows = [(system, key, c, float(lp[j])) for (key, codes), lp in zip(meta, lps) for j, c in enumerate(codes)]
    return pd.DataFrame(rows, columns=["system", "key", "code", "llm_lp"])


def neighbour_examples(mem, mem_keys, qenc, model: str, qkeys, desc_of: dict, system: str,
                       k: int = 5, loo: bool = True) -> dict:
    """key -> [(train term, "CODE desc")] for the k most similar labelled training terms
    (dense similarity; the query's own row is excluded when loo=True)."""
    import numpy as np

    from .features import clean_entity

    texts = [clean_entity(e)[0] for e in qkeys["entity"]]
    Q = qenc(model, texts)
    M = mem.dense[model]
    pos = {kk: i for i, kk in enumerate(mem_keys)}
    out = {}
    for s0 in range(0, len(Q), 2048):
        S = Q[s0 : s0 + 2048] @ M.T
        for r in range(len(S)):
            key = qkeys["key"].values[s0 + r]
            if loo and key in pos:
                S[r, pos[key]] = -9
        top = np.argsort(-S, axis=1)[:, :k]
        for r in range(len(S)):
            key = qkeys["key"].values[s0 + r]
            ex = []
            for j in top[r]:
                c = mem.codes[j]
                d = desc_of.get(c, c)
                ex.append((mem.texts[j], f"{c} {d}" if system == "ICD10CM" else d))
            out[key] = ex
    return out
