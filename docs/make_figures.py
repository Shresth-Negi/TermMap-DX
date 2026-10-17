"""Render the architecture diagram and result charts used in the write-up."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
INK, MUTED, A1, A2, A3, BG = "#1f2933", "#52606d", "#2563eb", "#0f766e", "#b45309", "#f5f7fa"


def box(ax, x, y, w, h, title, body="", color=A1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc="white", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top", fontsize=8.6, weight="bold", color=color)
    if body:
        ax.text(x + w / 2, y + h - 0.068, body, ha="center", va="top", fontsize=6.6, color=INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))


def architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    box(ax, 0.02, 0.80, 0.22, 0.16, "Input", "entity (verbatim)\nchunk text\nsubconcept (or inferred)", INK)
    box(ax, 0.29, 0.78, 0.30, 0.18, "1  Normalise",
        "html / markup, '(hcc)', bullets\nICD codes written in text\nabbreviation variants (SNOMED-mined\n+ curated lexicon)  ·  negation / history cues")
    box(ax, 0.64, 0.78, 0.34, 0.18, "SNOMED synonym bridge (ICD only)",
        "top SNOMED concepts' preferred terms\nbecome extra ICD query variants", A2)
    box(ax, 0.02, 0.47, 0.96, 0.25, "2  Candidate generation (per system, union of retrievers, ~60-120 codes)", "", A1)
    xs = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    labels = [("exact key", "normalised\ndescription match"), ("char TF-IDF", "3-4 grams:\nOCR / typos"),
              ("word TF-IDF", "rare exact\ntokens"), ("dense", "SapBERT · gte-base\n· Qwen3-Embedding"),
              ("training memory", "nearest coded terms\n(local convention)"), ("code in text", "e.g. 'oa knee m17.12'")]
    for x, (t, b) in zip(xs, labels):
        box(ax, x, 0.495, 0.14, 0.15, t, b, MUTED)
    box(ax, 0.02, 0.24, 0.45, 0.18, "3  Stage-1 ranker (LightGBM LambdaRank)",
        "~80 features per (term, code): every retriever's score for every\ncandidate, fuzzy match, token coverage, laterality, numbers / stage,\n"
        "negation, specificity (billable / unspecified / other / 7th char),\nsemantic tag x entity type, priors, neighbour + category votes", A1)
    box(ax, 0.52, 0.24, 0.46, 0.18, "4  Stage-2 ranker over top-20 shortlist",
        "stage-1 score / rank / gap\n+ cross-system coupling: ICD<->SNOMED co-occurrence (LOO)\n  and similarity to the other system's shortlist\n"
        "+ LLM listwise log-probs (Qwen2.5, local, 5 retrieved examples)", A3)
    box(ax, 0.25, 0.02, 0.50, 0.15, "Output (both systems, every row)",
        "ICD-10-CM code  ·  SNOMED CT concept\ncalibrated confidence (isotonic on OOF margin)  ·  top-5 alternatives", INK)
    arrow(ax, 0.24, 0.88, 0.29, 0.88); arrow(ax, 0.59, 0.87, 0.64, 0.87)
    arrow(ax, 0.44, 0.78, 0.44, 0.72); arrow(ax, 0.81, 0.78, 0.81, 0.72)
    arrow(ax, 0.25, 0.47, 0.25, 0.42); arrow(ax, 0.47, 0.33, 0.52, 0.33)
    arrow(ax, 0.75, 0.24, 0.6, 0.17)
    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight")


def results(report_paths):
    rows = []
    for name, p in report_paths:
        r = json.loads(Path(p).read_text())
        rows.append((name, r["ICD10CM"]["stage2_cv_top1"], r["SNOMEDCT_US"]["stage2_cv_top1"]))
    fig, ax = plt.subplots(figsize=(7, 3))
    import numpy as np
    x = np.arange(len(rows)); w = 0.38
    b1 = ax.bar(x - w / 2, [r[1] for r in rows], w, label="ICD-10-CM", color=A1)
    b2 = ax.bar(x + w / 2, [r[2] for r in rows], w, label="SNOMED CT", color=A2)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("5-fold CV exact match")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "ablation.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    architecture()
    if len(sys.argv) > 1:
        pairs = [a.split("=", 1) for a in sys.argv[1:]]
        results(pairs)
    print("figures written to", OUT)
