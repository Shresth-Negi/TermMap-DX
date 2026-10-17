"""Error analysis over out-of-fold (cross-validated) predictions.

Writes outputs/error_analysis<tag>.json and .md with:
  * exact-match accuracy per system (headline = min of the two)
  * ICD-10-CM partial credit (same 3-character category / same chapter)
  * long-tail accuracy by how often the gold code occurs elsewhere in training
  * failure-mode breakdown (retrieval miss, specificity, laterality, wrong concept, ...)
  * coverage / abstention curve from the stage-2 margin
  * sample errors per failure mode

Usage: python scripts/error_analysis.py --tag _v1
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from termmap.corpus import SYSTEMS, description_map, load_entries
from termmap.features import lat_of
from termmap.paths import ARTIFACTS, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

report, md = {}, [f"# Error analysis (out-of-fold, tag `{args.tag or 'default'}`)\n"]
for sys_ in SYSTEMS:
    o = pd.read_parquet(ARTIFACTS / f"oof_{sys_}{args.tag}.parquet")
    e = load_entries(sys_)
    desc = pd.Series(description_map(sys_))
    sem = e.drop_duplicates("code").set_index("code")["semtag"] if "semtag" in e else None
    freq = Counter(o["code"])
    o["gold_freq_loo"] = o["code"].map(freq) - 1
    o["gold_desc"] = o["code"].map(desc)
    o["pred_desc"] = o["pred"].map(desc)

    def mode(r):
        if r.ok:
            return "correct"
        if not r.in_shortlist:
            return "retrieval_miss"
        gl, pl = lat_of(str(r.gold_desc).lower()), lat_of(str(r.pred_desc).lower())
        if gl != pl and (gl or pl):
            return "laterality"
        if sys_ == "ICD10CM":
            if str(r.pred)[:3] == str(r.code)[:3]:
                return "specificity (same 3-char category)"
            if str(r.pred)[:1] == str(r.code)[:1]:
                return "wrong concept, same chapter"
            return "wrong concept, other chapter"
        if sem is not None and sem.get(r.pred) != sem.get(r.code):
            return "semantic-type mismatch (e.g. finding vs procedure)"
        return "wrong concept, same semantic type"

    o["mode"] = o.apply(mode, axis=1)
    acc = float(o["ok"].mean())
    r = {"n": int(len(o)), "exact_match": round(acc, 4)}
    if sys_ == "ICD10CM":
        r["same_3char_category"] = round(float((o["pred"].str[:3] == o["code"].str[:3]).mean()), 4)
        r["same_chapter_letter"] = round(float((o["pred"].str[:1] == o["code"].str[:1]).mean()), 4)
    bins = pd.cut(o["gold_freq_loo"], [-1, 0, 2, 9, 10**6], labels=["unseen (0)", "rare (1-2)", "mid (3-9)", "frequent (10+)"])
    lt = o.groupby(bins, observed=False)["ok"].agg(["mean", "size"]).round(4)
    r["long_tail"] = {str(k): {"acc": float(v["mean"]), "n": int(v["size"])} for k, v in lt.iterrows()}
    r["long_tail_acc_unseen_or_rare"] = round(float(o[o.gold_freq_loo <= 2]["ok"].mean()), 4)
    fm = o["mode"].value_counts()
    r["failure_modes"] = {k: {"n": int(v), "share_of_rows": round(v / len(o), 4)} for k, v in fm.items()}
    r["by_subconcept"] = o.groupby("subconcept")["ok"].agg(["mean", "size"]).round(4).to_dict(orient="index")
    # coverage / abstention: keep the most confident fraction
    oc = o.sort_values("margin", ascending=False).reset_index(drop=True)
    cov = {}
    for c in (1.0, 0.97, 0.9, 0.8, 0.7, 0.5):
        k = int(round(c * len(oc)))
        cov[f"{int(c*100)}%"] = round(float(oc["ok"].iloc[:k].mean()), 4)
    r["accuracy_at_coverage"] = cov
    if "pred_s1" in o:
        r["stage1_only_exact_match"] = round(float((o["pred_s1"] == o["code"]).mean()), 4)
    report[sys_] = r

    md.append(f"\n## {sys_}  (n={len(o)})\n")
    md.append(f"* exact match: **{acc:.4f}**" + (f"  (stage-1 only: {r.get('stage1_only_exact_match')})" if 'stage1_only_exact_match' in r else ""))
    if sys_ == "ICD10CM":
        md.append(f"* same 3-char category: {r['same_3char_category']}, same chapter: {r['same_chapter_letter']}")
    md.append("\n| gold-code frequency elsewhere in train | accuracy | n |\n|---|---|---|")
    for k, v in r["long_tail"].items():
        md.append(f"| {k} | {v['acc']:.4f} | {v['n']} |")
    md.append("\n| failure mode | rows | share |\n|---|---|---|")
    for k, v in r["failure_modes"].items():
        md.append(f"| {k} | {v['n']} | {v['share_of_rows']:.4f} |")
    md.append("\n| coverage (most confident) | accuracy |\n|---|---|")
    for k, v in cov.items():
        md.append(f"| {k} | {v:.4f} |")
    md.append("\n| subconcept | accuracy | n |\n|---|---|---|")
    for k, v in r["by_subconcept"].items():
        md.append(f"| {k} | {v['mean']:.4f} | {int(v['size'])} |")
    md.append("\nSample errors:\n")
    for m_, g in o[~o.ok].groupby("mode"):
        md.append(f"\n**{m_}**\n")
        for _, x in g.sample(min(6, len(g)), random_state=0).iterrows():
            md.append(f"- `{x.entity}` → gold `{x.code}` {x.gold_desc} | pred `{x.pred}` {x.pred_desc}")
    o.to_parquet(ARTIFACTS / f"errors_{sys_}{args.tag}.parquet")

# ---- label consistency: an empirical ceiling for exact match under the current gold labels
from termmap.features import clean_entity
from termmap.io import load_train
from termmap.lexical import LexicalIndex
import re as _re

def _canon(t):
    t = _re.sub(r"[^a-z0-9 ]", " ", clean_entity(t)[0])
    return " ".join(sorted(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in t.split()))

tr = load_train()
md.append("\n## Label consistency (empirical ceiling)\n")
md.append("Pairs of training terms that are the same after normalisation (case, punctuation, plural, word order), "
          "or nearly identical (char 3-4-gram cosine >= 0.95), and how often their gold codes agree:\n")
md.append("| system | canonical-duplicate pairs | same gold code | near-identical pairs (cos>=0.95) | same gold code |\n|---|---|---|---|---|")
for sys_ in SYSTEMS:
    q = tr[tr.coding_system == sys_].reset_index(drop=True)
    q["k"] = q.entity.map(_canon)
    pairs = agree = 0
    for _, d in q.groupby("k"):
        c = d.code.tolist()
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                pairs += 1; agree += c[i] == c[j]
    texts = [clean_entity(e)[0] for e in q.entity]
    ids, sc = LexicalIndex("char", "x").build(texts).search(texts, k=2)
    near = [(i, ids[i][1]) for i in range(len(q)) if ids[i][1] >= 0 and sc[i][1] >= 0.95]
    nagree = float(np.mean([q.code[i] == q.code[j] for i, j in near])) if near else float("nan")
    report[sys_]["label_consistency"] = {"canonical_pairs": pairs, "canonical_agree": round(agree / max(pairs, 1), 4),
                                         "near_pairs": len(near), "near_agree": round(nagree, 4)}
    md.append(f"| {sys_} | {pairs} | {agree / max(pairs, 1):.3f} | {len(near)} | {nagree:.3f} |")

report["headline_min_exact_match"] = min(report[s]["exact_match"] for s in SYSTEMS)
(OUT / f"error_analysis{args.tag}.json").write_text(json.dumps(report, indent=2))
(OUT / f"error_analysis{args.tag}.md").write_text("\n".join(md))
print(json.dumps({s: {k: v for k, v in report[s].items() if k in ("exact_match", "long_tail", "failure_modes", "accuracy_at_coverage")}
                  for s in SYSTEMS}, indent=1))
