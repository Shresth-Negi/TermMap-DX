"""Trained artifacts on the Hugging Face Hub: which files a model set needs, and upload / download.

Everything else the engine uses is either public (encoders and LLMs, pulled from the Hub by name),
supplied by the challenge (training file, dictionaries in data/raw), or rebuilt deterministically
on first use (dictionary caches, lexical indexes, training-memory embeddings, subconcept model).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .corpus import SYSTEMS
from .paths import ARTIFACTS

DEFAULT_REPO = "Rohan-Ark/TermMap-DX"
SETS = ("_d3llm14e4", "_d3llm7e4", "_d3ce4", "_sap4")
MANIFEST = "manifest.json"


def repo_id() -> str:
    return os.environ.get("TERMMAP_HF_REPO", DEFAULT_REPO)


def _config(tag: str, root: Path) -> dict:
    return json.loads((root / f"config{tag}.json").read_text())


def set_files(tag: str, root: Path = ARTIFACTS) -> list[str]:
    """Relative paths needed to run model set `tag` and every set it falls back to."""
    files: list[str] = []
    seen: set[str] = set()
    while tag and tag not in seen:
        seen.add(tag)
        cfg = _config(tag, root)
        files.append(f"config{tag}.json")
        for s in SYSTEMS:
            files.append(f"calib_{s}{tag}.json")
            for st in ("_s1", "_s2"):
                files += [f"ranker_{s}{st}{tag}.txt", f"ranker_{s}{st}{tag}.json"]
            files += [f"emb_{m}_{s}.npy" for m in cfg["dense"]]
        for name in cfg.get("extra", []):
            if Path(name).name.startswith("ce_scores_train"):
                t = Path(name).name[len("ce_scores_train"):-len(".parquet")]
                for s in SYSTEMS:
                    d = root / f"ce_{s}{t}"
                    files += [f"ce_{s}{t}/{f.name}" for f in sorted(d.iterdir()) if f.is_file()]
        tag = cfg.get("fallback", "")
    return list(dict.fromkeys(files))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def stage(out: Path, sets=SETS, card: str = "") -> list[str]:
    """Copy the artifacts of `sets` into `out` (hard links when possible) with a sha256 manifest."""
    out.mkdir(parents=True, exist_ok=True)
    files = list(dict.fromkeys(f for t in sets for f in set_files(t)))
    manifest = {"sets": list(sets), "files": {}}
    for rel in files:
        src, dst = ARTIFACTS / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        manifest["files"][rel] = {"sha256": _sha256(dst), "bytes": dst.stat().st_size}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    if card:
        (out / "README.md").write_text(card)
    return files


def _set_names(tag: str, cfg: dict) -> list[str]:
    names = [f"config{tag}.json"]
    for s in SYSTEMS:
        names.append(f"calib_{s}{tag}.json")
        for st in ("_s1", "_s2"):
            names += [f"ranker_{s}{st}{tag}.txt", f"ranker_{s}{st}{tag}.json"]
        names += [f"emb_{m}_{s}.npy" for m in cfg["dense"]]
    return names


def download(sets=("_d3llm14e4",), repo: str | None = None) -> list[str]:
    """Fetch the files of `sets` (and of the sets they fall back to) straight into ARTIFACTS,
    skipping files already present with the right checksum. Returns the files downloaded."""
    from huggingface_hub import hf_hub_download

    repo = repo or repo_id()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(hf_hub_download(repo, MANIFEST, local_dir=ARTIFACTS)).read_text())["files"]
    want: list[str] = []
    for tag in sets:
        while tag and f"config{tag}.json" not in want:
            cfg = json.loads(Path(hf_hub_download(repo, f"config{tag}.json", local_dir=ARTIFACTS)).read_text())
            want += _set_names(tag, cfg)
            for name in cfg.get("extra", []):
                if name.startswith("ce_scores_train"):
                    t = name[len("ce_scores_train"):-len(".parquet")]
                    want += [f for f in man if f.startswith(tuple(f"ce_{s}{t}/" for s in SYSTEMS))]
            tag = cfg.get("fallback", "")
    missing = [f for f in want if f not in man]
    if missing:
        raise RuntimeError(f"{repo} lacks {missing}")
    got = []
    for rel in dict.fromkeys(want):
        dst = ARTIFACTS / rel
        if dst.exists() and _sha256(dst) == man[rel]["sha256"]:
            continue
        hf_hub_download(repo, rel, local_dir=ARTIFACTS)
        if _sha256(dst) != man[rel]["sha256"]:
            raise RuntimeError(f"checksum mismatch for {rel}")
        got.append(rel)
    return got
