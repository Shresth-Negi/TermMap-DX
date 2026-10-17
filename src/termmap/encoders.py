"""Dense sentence encoders used for retrieval.

Every encoder returns L2-normalised float32 vectors so inner product = cosine.
Model weights are pinned by Hugging Face repo id + revision (see MODELS).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# name -> (hf repo id, pooling, max_len, query prefix). Documents never get a prefix.
MODELS = {
    "sapbert": ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "cls", 32, ""),
    "gte": ("thenlper/gte-base", "mean", 48, ""),
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", "last", 64,
              "Instruct: Given a clinical term extracted from a medical chart, retrieve the medical "
              "concept name it refers to\nQuery:"),
}


class Encoder:
    def __init__(self, name: str, device: str | None = None, path: str | None = None):
        repo, self.pooling, self.max_len, self.prefix = MODELS[name]
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        src = path or repo
        self.tok = AutoTokenizer.from_pretrained(src)
        if self.pooling == "last":
            self.tok.padding_side = "left"
        self.model = AutoModel.from_pretrained(src).to(self.device).eval()
        if self.device == "cuda":  # Qwen3 is trained in bf16; BERT-style encoders are fine in fp16
            self.model = self.model.to(torch.bfloat16 if self.pooling == "last" else torch.float16)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 512, show: bool = False, query: bool = False) -> np.ndarray:
        pre = self.prefix if query else ""
        texts = [pre + (t or "").lower() for t in texts]
        order = np.argsort([len(t) for t in texts])  # length bucketing: far less padding
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for b, s in enumerate(range(0, len(texts), batch_size)):
            idx = order[s : s + batch_size]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                           max_length=self.max_len + (48 if pre else 0), return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            if self.pooling == "cls":
                v = h[:, 0]
            elif self.pooling == "last":
                v = h[:, -1]  # left padding: last position is the final real token
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out[idx] = v.cpu().numpy()
            if show and b % 200 == 0:
                print(f"[{self.name}] {s}/{len(texts)}", flush=True)
        return out


def set_threads():
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    torch.set_num_threads(n)
