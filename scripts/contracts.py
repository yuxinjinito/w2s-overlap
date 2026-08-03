#!/usr/bin/env python3
"""The contracts every layer agrees on.

The two shared row types, the custom-score slot names, and the per-arm seed
offsets. This module depends on nothing in the project, so anything may import
it; nothing here may ever import upward.

The seed table exists because the offsets used to live as scattered literals
and two arm families ended up sharing a stream: l2robust and mlpstep1 both sat
on 27000/27500, so their balancing draws were identical whenever both ran in
one sweep. mlpstep1 keeps the old values (it is the active line and its past
runs stay reproducible); l2robust, a closed negative, moved to 35000. orz
"""
from dataclasses import dataclass

from datasets import Dataset


@dataclass
class LoraExample:
    id: str
    source_id: str
    text: str
    label: int
    answer_suffix: str

    @property
    def prompt(self) -> str:
        return f"{self.text}{self.answer_suffix}"


@dataclass(frozen=True)
class SplitBundle:
    weak_train: Dataset
    strong_train: Dataset
    val: Dataset
    test: Dataset


CUSTOM_SLOTS = ("cs1", "cs2", "cs3", "cs4")

# Every RNG stream is args.seed + offset (+ fi for per-keep-fraction arms,
# + idx / + 100*sidx + i / + 1000*slot for the indexed families, noted inline).
# One entry per stream that exists; the uniqueness assert below is the whole
# point, so add new arms here first and let a collision fail at import time.
SEED_OFFSETS = {
    "weak_label_balanced": 500,
    "weak_correct_balanced": 600,
    "random_balanced": 1000,        # + idx per control arm
    "confidence_middle": 2000,
    "confidence_high": 3000,
    "knn_middle": 4000,
    "knn_mixed": 5000,
    "knn_high": 5500,
    "random_unbalanced": 6000,      # + idx per control arm
    "committee_agree": 7000,
    "committee_disagree": 8000,
    "committee_fit": 9000,
    "kf_committee_agree": 10000,
    "kf_committee_disagree": 11000,
    "kf_committee_top": 12000,
    "kf_random_matched": 12500,
    "kf_knn_high": 13000,
    "kf_knn_low": 13500,
    "kf_confidence_high": 14000,
    "kf_confidence_low": 14500,
    "kf_confidence_cleanhard": 14700,
    "kf_knn_mixed": 15000,
    "rp_high": 16000,
    "rp_low": 16500,
    "el_high": 17000,
    "el_low": 17500,
    "conf_entropy_high": 18000,
    "conf_entropy_low": 18500,
    "agree_full": 19000,
    "disagree_full": 19500,
    "quantile": 20000,              # + 100*sidx + bin
    "mlpalign_high": 21000,
    "mlpalign_low": 21500,
    "cca_high": 22000,
    "cca_low": 22500,
    "l2resid_high": 23000,
    "l2resid_low": 23500,
    "rp_conf_high": 24000,
    "mlprp_high": 25000,
    "mlprp_low": 25500,
    "ccarobust_high": 26000,
    "ccarobust_low": 26500,
    "mlpstep1_high": 27000,
    "mlpstep1_low": 27500,
    "lda_high": 28000,
    "lda_low": 28500,
    "linstep1_high": 29000,
    "linstep1_low": 29500,
    "softrp_high": 30000,
    "softrp_low": 30500,
    "custom_high": 31000,           # + 1000*slot, slots 0..3 reach 34000
    "custom_low": 31500,            # + 1000*slot, slots 0..3 reach 34500
    "l2robust_high": 35000,
    "l2robust_low": 35500,
    "purity_flip": 36000,           # + target percent per grid point
    "purity_iid": 37000,            # + target percent + 200*draw
}

_vals = list(SEED_OFFSETS.values())
assert len(set(_vals)) == len(_vals), "seed offsets collide; pick a fresh stream"
