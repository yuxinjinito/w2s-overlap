#!/usr/bin/env python3
"""Systematic Stage-0 candidate scout for W2S testbeds.

Sweeps the HF Hub for NLI-family datasets (search terms + the tasksource collection),
then applies cheap structural filters WITHOUT downloading full data (streaming a handful
of rows): English, premise/hypothesis-style fields, 3-way label space, >= min_rows.
Emits a ranked candidate list for the Stage-1.5 trio screen.

This replaces hand-picked Stage-0 pools (3 rounds, 15 candidates, 1 survivor) with a
systematic sweep; the known failure modes are handled downstream by the trio screen
(base-too-strong / recall via 7B native+discrim, weak-noise + teacher-ceiling via the
0.5B probe).
"""
from __future__ import annotations

import argparse
import itertools
import json

from huggingface_hub import list_datasets

PAIR_FIELDS = [("premise", "hypothesis"), ("sentence1", "sentence2"), ("text_a", "text_b"),
               ("evidence", "claim"), ("context", "hypothesis")]
LABEL_FIELDS = ["label", "gold_label", "labels", "gold"]
SKIP_SUBSTR = ["translat", "multiling", "xnli", "-fr", "-de", "-es", "-zh", "-tr", "-ru",
               "korean", "japanese", "arabic", "indic", "swahili", "turkish", "persian",
               "sst", "sentiment", "paraphrase-detection"]
ALREADY = {"anli", "wanli", "control", "snli", "mnli", "multi_nli", "fever", "nli_fever",
           "lingnli", "conjnli", "imppres", "robust_nli", "vitaminc", "sick", "scitail",
           "hans", "wnli", "rte", "dnc"}


def looks_english_3way(ds_id: str, max_rows: int = 40):
    """Stream a few rows; return (ok, reason, meta)."""
    from datasets import load_dataset

    try:
        ds = load_dataset(ds_id, split="train", streaming=True)
        rows = list(itertools.islice(ds, max_rows))
    except Exception:
        for split in ("validation", "test", "dev"):
            try:
                ds = load_dataset(ds_id, split=split, streaming=True)
                rows = list(itertools.islice(ds, max_rows))
                break
            except Exception:
                rows = []
        if not rows:
            return False, "unloadable/config-required", {}
    if not rows:
        return False, "empty", {}
    cols = set(rows[0].keys())
    pair = next(((a, b) for a, b in PAIR_FIELDS if a in cols and b in cols), None)
    if pair is None:
        return False, f"no premise/hypothesis fields ({sorted(cols)[:6]})", {}
    lab = next((f for f in LABEL_FIELDS if f in cols), None)
    if lab is None:
        return False, "no label field", {}
    vals = {str(r.get(lab)) for r in rows}
    if not (2 < len(vals) <= 4):
        return False, f"label space {sorted(vals)[:5]} not 3-way", {}
    text = " ".join(str(rows[0][pair[0]]) + str(rows[0][pair[1]]) for _ in [0])[:200]
    if sum(c.isascii() for c in text) / max(1, len(text)) < 0.95:
        return False, "non-English sample", {}
    return True, "ok", {"pair": pair, "label": lab, "label_vals": sorted(vals)[:4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-check", type=int, default=60, help="max datasets to schema-check")
    ap.add_argument("--out", default="candidate_scout_report.json")
    args = ap.parse_args()

    seen: dict[str, str] = {}
    queries = ["adversarial nli", "nli stress", "natural language inference hard",
               "entailment adversarial", "logical nli", "presupposition nli",
               "counterfactual nli", "contrast set", "scientific nli", "legal nli",
               "contract nli", "clinical nli", "defeasible inference", "dynabench"]
    cands: list[str] = []
    for q in queries:
        for d in list_datasets(search=q, limit=40):
            cands.append(d.id)
    for d in list_datasets(author="tasksource", limit=300):
        if "nli" in d.id.lower() or "entail" in d.id.lower():
            cands.append(d.id)

    ordered = []
    for cid in cands:
        low = cid.lower()
        if cid in ordered or any(s in low for s in SKIP_SUBSTR):
            continue
        if any(a in low.split("/")[-1] for a in ALREADY):
            seen[cid] = "already screened/used"
            continue
        ordered.append(cid)

    print(f"scout: {len(ordered)} unique candidates after name filters; schema-checking top {args.max_check}")
    report = []
    for cid in ordered[: args.max_check]:
        ok, reason, meta = looks_english_3way(cid)
        report.append({"id": cid, "pass": ok, "reason": reason, **meta})
        mark = "PASS" if ok else "    "
        print(f"  [{mark}] {cid}: {reason}")
    passing = [r for r in report if r["pass"]]
    print(f"\n=== {len(passing)} structural passes -> trio-screen queue ===")
    for r in passing:
        print(f"  {r['id']}  fields={r.get('pair')} labels={r.get('label_vals')}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
