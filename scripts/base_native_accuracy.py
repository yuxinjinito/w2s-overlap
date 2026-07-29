#!/usr/bin/env python3
"""Native (likelihood) vs discrimination (yes/no) multiple-choice accuracy of a base model.

A low discrimination-format accuracy can mean either (a) the model does not know the task,
or (b) the untuned model simply does not follow the yes/no format. This scores each option
two ways and reports both, so the gap tells you which:

  - NATIVE        : length-normalised log P(option | context), argmax over options.
  - DISCRIMINATION: P(" yes") for "{context} {option}{suffix}", argmax of P(yes) over options
                    (this matches how the pipeline evaluates the base model).

If NATIVE >> DISCRIMINATION, the model knows the task and the low discrimination accuracy is
a format limitation -- so a weak-to-strong "lift" from training partly reflects the model
learning the format / re-surfacing pretrained knowledge, not new capability.

Usage (on a GPU box):
  PYTHONPATH=scripts python3 scripts/base_native_accuracy.py \
      --dataset hellaswag --model meta-llama/Llama-3.1-8B --n-questions 500 --device cuda
"""
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dream_w2s_baselines import resolve_dtype, score_candidates

DEFAULT_SUFFIX = " Is the candidate answer correct? Answer:"


def hellaswag_questions(n: int, seed: int):
    raw = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        if ex["label"] == "":  # hidden test labels
            continue
        out.append(((ex.get("ctx") or "").strip(), [e.strip() for e in ex["endings"]], int(ex["label"])))
        if len(out) >= n:
            break
    return out


def dream_questions(n: int, seed: int):
    import random as _random

    from run_dream_paper_linear_probe import iter_dream_question_rows, load_dream_raw

    rows = list(iter_dream_question_rows(load_dream_raw("test")))
    _random.Random(seed).shuffle(rows)
    out = []
    for ex in rows:
        joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else str(ex["dialogue"])
        ctx = f"{joined}\n\nQ: {ex['question']} A:"
        choices = list(ex["choice"])
        out.append((ctx, choices, choices.index(ex["answer"])))
        if len(out) >= n:
            break
    return out


ANLI_RELATIONS = ["entailment", "neutral", "contradiction"]


def anli_questions(n: int, seed: int, anli_round: str = "r2"):
    raw = load_dataset("facebook/anli", split=f"test_{anli_round}").shuffle(seed=seed)
    out = []
    for ex in raw:
        gold = int(ex["label"])
        if gold not in (0, 1, 2):
            continue
        ctx = (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


def sciq_questions(n: int, seed: int):
    raw = load_dataset("allenai/sciq", split="test").shuffle(seed=seed)
    out = []
    for ex in raw:
        ctx = f"Q: {ex['question']} A:"
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        out.append((ctx, options, 0))
        if len(out) >= n:
            break
    return out


# --- candidate W2S testbeds (Stage-1 headroom screen): adversarial NLI + logical-reasoning MC.
# All return (ctx, options, gold_idx), like the loaders above. Chosen because difficulty is from
# adversarial construction / reasoning (latent, elicitable capability), so they are plausibly
# base-near-chance AND robust to pretraining contamination (see dataset_collection_method note).

def wanli_questions(n: int, seed: int):
    """WANLI: worker-and-AI adversarial NLI (3-way relation), same format as ANLI."""
    raw = load_dataset("alisawuffles/WANLI", split="test").shuffle(seed=seed)
    rel2idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
    out = []
    for ex in raw:
        gold = rel2idx.get(ex["gold"])
        if gold is None:
            continue
        ctx = (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


def reclor_questions(n: int, seed: int):
    """ReClor: LSAT/GMAT logical-reasoning MC (4 options). Test labels hidden -> use validation."""
    raw = load_dataset("metaeval/reclor", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        ctx = f"{ex['context'].strip()}\nQ: {ex['question'].strip()} A:"
        out.append((ctx, [str(a).strip() for a in ex["answers"]], int(ex["label"])))
        if len(out) >= n:
            break
    return out


def art_questions(n: int, seed: int):
    """ART / alphaNLI: abductive NLI -- pick the hypothesis that better explains the observations
    (2 options). label is 1-indexed (1 or 2)."""
    raw = load_dataset("allenai/art", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        ctx = (
            f"Observation 1: {ex['observation_1'].strip()}\n"
            f"Observation 2: {ex['observation_2'].strip()}\n"
            "Q: Which hypothesis better explains the observations? A:"
        )
        options = [ex["hypothesis_1"].strip(), ex["hypothesis_2"].strip()]
        out.append((ctx, options, int(ex["label"]) - 1))
        if len(out) >= n:
            break
    return out


def logiqa2_questions(n: int, seed: int):
    """LogiQA 2.0: logical-reasoning MC (4 options). Rows are a JSON string under 'text'.
    This HF mirror mixes in NLI-style rows (no options); keep only the MC rows."""
    import json as _json

    raw = load_dataset("datatune/LogiQA2.0", split="test").shuffle(seed=seed)
    out = []
    for row in raw:
        try:
            ex = _json.loads(row["text"])
        except Exception:
            continue
        if not all(k in ex for k in ("options", "answer", "question", "text")):
            continue  # skip the mixed-in NLI-style rows
        options = [str(o).strip() for o in ex["options"]]
        try:
            gold = int(ex["answer"])
        except (ValueError, TypeError):
            continue
        if not 0 <= gold < len(options):
            continue
        ctx = f"{ex['text'].strip()}\nQ: {ex['question'].strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def _nli_relation_questions(hf: str, split: str, premise_key: str, hypothesis_key: str,
                            label_key: str, label_map: dict, n: int, seed: int):
    """Generic 3-way NLI loader -> (ctx, [entailment,neutral,contradiction], gold). Used for the
    adversarial-NLI low-base candidates (ConTRoL, SNLI-hard)."""
    raw = load_dataset(hf, split=split).shuffle(seed=seed)
    out = []
    for ex in raw:
        gold = label_map.get(ex[label_key])
        if gold is None:
            continue
        ctx = (
            f"Premise: {str(ex[premise_key]).strip()}\n"
            f"Hypothesis: {str(ex[hypothesis_key]).strip()}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


_NLI3 = {"entailment": 0, "neutral": 1, "contradiction": 2}


def control_questions(n: int, seed: int):
    """ConTRoL: contextual-reasoning adversarial NLI (3-way), long premises."""
    return _nli_relation_questions("tasksource/ConTRoL-nli", "test", "premise", "hypothesis",
                                   "label", _NLI3, n, seed)


def snli_hard_questions(n: int, seed: int):
    """SNLI hard subset (3-way NLI); drops no-consensus '-' labels via the map."""
    return _nli_relation_questions("au123/snli-hard", "test", "sentence1", "sentence2",
                                   "gold_label", _NLI3, n, seed)


def vitaminc_questions(n: int, seed: int):
    """VitaminC: contrastive/adversarial fact verification (3-way), evidence + claim."""
    raw = load_dataset("tals/vitaminc", split="test").shuffle(seed=seed)
    rel = {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}
    opts = ["supported", "not enough info", "refuted"]
    out = []
    for ex in raw:
        gold = rel.get(ex["label"])
        ev = str(ex["evidence"]).strip()
        if gold is None or not ev:
            continue
        ctx = (
            f"Evidence: {ev}\n"
            f"Claim: {str(ex['claim']).strip()}\n"
            "Q: Based on the evidence, the claim is? A:"
        )
        out.append((ctx, list(opts), gold))
        if len(out) >= n:
            break
    return out


def codah_questions(n: int, seed: int):
    """CODAH: adversarially-filtered commonsense sentence completion (4 options).
    Only a train split exists on HF; fine for a base screen."""
    raw = load_dataset("codah", "codah", split="train").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(o).strip() for o in ex["candidate_answers"]]
        gold = int(ex["correct_answer_idx"])
        if not 0 <= gold < len(options):
            continue
        ctx = f"Q: Complete this naturally: {str(ex['question_propmt']).strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def cosmosqa_questions(n: int, seed: int):
    """Cosmos QA: commonsense reading comprehension (4 options). Test split is
    unlabeled, so screen on validation."""
    raw = load_dataset("cosmos_qa", "default", split="validation",
                       revision="refs/convert/parquet").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(ex[f"answer{i}"]).strip() for i in range(4)]
        gold = int(ex["label"])
        if not 0 <= gold < 4:
            continue
        ctx = f"{str(ex['context']).strip()}\nQ: {str(ex['question']).strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def musr_questions(n: int, seed: int):
    """MuSR: multistep soft reasoning over long narratives (murder mysteries split;
    'choices' is a stringified python list)."""
    import ast as _ast

    raw = load_dataset("TAUR-Lab/MuSR", split="murder_mysteries").shuffle(seed=seed)
    out = []
    for ex in raw:
        try:
            options = [str(o).strip() for o in _ast.literal_eval(ex["choices"])]
        except (ValueError, SyntaxError):
            continue
        gold = int(ex["answer_index"])
        if not 0 <= gold < len(options):
            continue
        ctx = f"{str(ex['narrative']).strip()}\nQ: {str(ex['question']).strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def quail_questions(n: int, seed: int):
    """QuAIL: multi-domain reading comprehension (4 options, incl. unanswerable)."""
    raw = load_dataset("textmachinelab/quail", "default", split="validation",
                       revision="refs/convert/parquet").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(o).strip() for o in ex["answers"]]
        gold = int(ex["correct_answer_id"])
        if not 0 <= gold < len(options):
            continue
        ctx = f"{str(ex['context']).strip()}\nQ: {str(ex['question']).strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def _letter_choices_questions(hf: str, config, split: str, n: int, seed: int,
                              q_prefix: str = ""):
    """Generic loader for the AllenAI letter-keyed MC schema: question + choices{label,text}
    + answerKey. Used by CommonsenseQA (5-way) and QASC (8-way)."""
    raw = (load_dataset(hf, config, split=split) if config else load_dataset(hf, split=split))
    raw = raw.shuffle(seed=seed)
    out = []
    for ex in raw:
        labels = list(ex["choices"]["label"])
        options = [str(t).strip() for t in ex["choices"]["text"]]
        key = str(ex["answerKey"]).strip()
        if key not in labels or not options:
            continue
        gold = labels.index(key)
        ctx = f"{q_prefix}Q: {str(ex['question']).strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def csqa_questions(n: int, seed: int):
    """CommonsenseQA: 5-way commonsense MC with adversarial ConceptNet distractors.
    Test split is unlabeled -> screen on validation."""
    return _letter_choices_questions("tau/commonsense_qa", None, "validation", n, seed)


def qasc_questions(n: int, seed: int):
    """QASC: 8-way science MC requiring two-fact composition. Test split is
    unlabeled -> screen on validation."""
    return _letter_choices_questions("allenai/qasc", None, "validation", n, seed)


def lingnli_questions(n: int, seed: int):
    """LingNLI: linguist-written adversarial NLI (3-way), validation split."""
    return _nli_relation_questions("tasksource/lingnli", "validation", "premise", "hypothesis",
                                   "label", _NLI3, n, seed)


def fevernli_questions(n: int, seed: int):
    """FEVER-NLI: fact-verification recast as 3-way (SUPPORTS/NEI/REFUTES), evidence + claim.
    dev split (test is unlabeled)."""
    raw = load_dataset("pietrolesci/nli_fever", split="dev").shuffle(seed=seed)
    rel = {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}
    opts = ["supported", "not enough info", "refuted"]
    out = []
    for ex in raw:
        gold = rel.get(str(ex["fever_gold_label"]))
        ev = str(ex["premise"]).strip()
        cl = str(ex["hypothesis"]).strip()
        if gold is None or not ev or not cl:
            continue
        ctx = f"Evidence: {ev}\nClaim: {cl}\nQ: Based on the evidence, the claim is? A:"
        out.append((ctx, list(opts), gold))
        if len(out) >= n:
            break
    return out


def imppres_questions(n: int, seed: int):
    """ImpPres (presupposition section): structurally adversarial 3-way NLI probing
    presupposition projection; merges several presupposition configs."""
    configs = ["presupposition_all_n_presupposition", "presupposition_both_presupposition",
               "presupposition_change_of_state", "presupposition_cleft_existence"]
    out = []
    per = max(1, n // len(configs) + 1)
    for cfg in configs:
        dd = load_dataset("facebook/imppres", cfg)
        raw = dd[list(dd.keys())[0]].shuffle(seed=seed)
        got = 0
        for ex in raw:
            lab = str(ex.get("gold_label", "")).strip().lower()
            try:
                gold = int(lab)  # numeric string convention: 0=ent 1=neu 2=con
            except ValueError:
                gold = _NLI3.get(lab)
            if gold not in (0, 1, 2):
                continue
            ctx = (
                f"Premise: {str(ex['premise']).strip()}\n"
                f"Hypothesis: {str(ex['hypothesis']).strip()}\n"
                "Q: What is the relationship from the premise to the hypothesis? A:"
            )
            out.append((ctx, list(ANLI_RELATIONS), gold))
            got += 1
            if got >= per or len(out) >= n:
                break
        if len(out) >= n:
            break
    return out[:n]


def conjnli_questions(n: int, seed: int):
    """ConjNLI: adversarial NLI over conjunctions (and/or/but/nor), 3-way. Loaded from the
    authors' GitHub TSV (no HF mirror exists)."""
    import random as _random

    url = "https://raw.githubusercontent.com/swarnaHub/ConjNLI/master/data/NLI/conj_dev.tsv"
    ds = load_dataset("csv", data_files={"dev": url}, delimiter="\t")["dev"]
    rows = list(ds)
    _random.Random(seed).shuffle(rows)
    out = []
    for ex in rows:
        keys = list(ex.keys())
        prem = str(ex.get("premise", ex.get(keys[0], ""))).strip()
        hyp = str(ex.get("hypothesis", ex.get(keys[1], ""))).strip()
        gold = _NLI3.get(str(ex.get("label", ex.get(keys[2], ""))).strip().lower())
        if gold is None or not prem or not hyp:
            continue
        ctx = (
            f"Premise: {prem}\nHypothesis: {hyp}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


def robustnli_questions(n: int, seed: int):
    """Robust-NLI stress-test collection (PI_CD split: counterfactual/distraction stress),
    3-way, adversarially constructed from MNLI."""
    raw = load_dataset("pietrolesci/robust_nli", split="PI_CD").shuffle(seed=seed)
    out = []
    for ex in raw:
        cols = {k.lower(): k for k in ex.keys()}
        prem = str(ex.get(cols.get("premise", "premise"), "")).strip()
        hyp = str(ex.get(cols.get("hypothesis", "hypothesis"), "")).strip()
        lab = str(ex.get(cols.get("label", "label"), "")).strip().lower()
        try:
            gold = int(lab)  # MNLI convention: 0=entailment 1=neutral 2=contradiction
        except ValueError:
            gold = _NLI3.get(lab)
        if gold not in (0, 1, 2) or not prem or not hyp:
            continue
        ctx = (
            f"Premise: {prem}\nHypothesis: {hyp}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


def dynasent_questions(n: int, seed: int):
    """DynaSent R2: Dynabench human-adversarial sentiment (3-way) -- the ANLI-style
    model-in-the-loop collection, non-NLI."""
    raw = load_dataset("parquet", data_files={
        "train": "hf://datasets/dynabench/dynasent@refs/convert/parquet/dynabench.dynasent.r2.all/train/*.parquet"
    }, split="train").shuffle(seed=seed)
    opts = ["positive", "negative", "neutral"]
    lab2idx = {"positive": 0, "negative": 1, "neutral": 2}
    out = []
    for ex in raw:
        gold = lab2idx.get(str(ex.get("gold_label", "")).strip().lower())
        sent = str(ex.get("sentence", "")).strip()
        if gold is None or not sent:
            continue
        ctx = f"Sentence: {sent}\nQ: What is the sentiment of the sentence? A:"
        out.append((ctx, list(opts), gold))
        if len(out) >= n:
            break
    return out


def quality_questions(n: int, seed: int):
    """QuALITY: human-written long-document MC (4 options), hard-annotated exam-style."""
    raw = load_dataset("emozilla/quality", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(o).strip() for o in ex.get("options", [])]
        try:
            gold = int(ex.get("answer", -1)) - 1  # 1-indexed
        except (TypeError, ValueError):
            continue
        art = str(ex.get("article", "")).strip()
        q = str(ex.get("question", "")).strip()
        if not (0 <= gold < len(options)) or not art or not q:
            continue
        ctx = f"{art[:4000]}\nQ: {q} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def race_questions(n: int, seed: int):
    """RACE (high): exam reading comprehension MC (4 options), human-written."""
    raw = load_dataset("ehovy/race", "high", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(o).strip() for o in ex.get("options", [])]
        gold = "ABCD".find(str(ex.get("answer", "")).strip())
        art = str(ex.get("article", "")).strip()
        q = str(ex.get("question", "")).strip()
        if not (0 <= gold < len(options)) or not art or not q:
            continue
        ctx = f"{art[:4000]}\nQ: {q} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def siqa_questions(n: int, seed: int):
    """Social IQa: social commonsense MC (3 options), adversarially filtered."""
    raw = load_dataset("allenai/social_i_qa", "default", split="validation",
                       revision="refs/convert/parquet").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(ex.get(k, "")).strip() for k in ("answerA", "answerB", "answerC")]
        try:
            gold = int(ex.get("label", 0)) - 1  # 1-indexed
        except (TypeError, ValueError):
            continue
        c = str(ex.get("context", "")).strip()
        q = str(ex.get("question", "")).strip()
        if not (0 <= gold < 3) or not c or not q:
            continue
        ctx = f"{c}\nQ: {q} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def mmlupro_questions(n: int, seed: int):
    """MMLU-Pro: 10-option MC with model-generated adversarial distractors (knowledge-heavy,
    but the only large >=10-way pool; high-way screen per the way-count relaxation)."""
    raw = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=seed)
    out = []
    for ex in raw:
        options = [str(o).strip() for o in ex.get("options", [])]
        gold = ex.get("answer_index", -1)
        q = str(ex.get("question", "")).strip()
        if not isinstance(gold, int) or not (0 <= gold < len(options)) or not q or len(options) < 6:
            continue
        ctx = f"Q: {q} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def aquarat_questions(n: int, seed: int):
    """AQuA-RAT: 5-option math word problems (large pool; weak-noise risk expected)."""
    raw = load_dataset("deepmind/aqua_rat", "raw", split="train").shuffle(seed=seed)
    out = []
    for ex in raw:
        opts_raw = [str(o).strip() for o in ex.get("options", [])]
        options = [o.split(")", 1)[1].strip() if ")" in o else o for o in opts_raw]
        gold = "ABCDE".find(str(ex.get("correct", "")).strip())
        q = str(ex.get("question", "")).strip()
        if not (0 <= gold < len(options)) or not q:
            continue
        ctx = f"Q: {q} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


def riddlesense_questions(n: int, seed: int):
    """RiddleSense: 5-option human-written riddles (adversarial commonsense)."""
    raw = load_dataset("INK-USC/riddle_sense", "default", split="validation",
                       revision="refs/convert/parquet").shuffle(seed=seed)
    out = []
    for ex in raw:
        labels = list(ex["choices"]["label"])
        options = [str(t).strip() for t in ex["choices"]["text"]]
        key = str(ex["answerKey"]).strip()
        if key not in labels or not options:
            continue
        out.append((f"Q: {str(ex['question']).strip()} A:", options, labels.index(key)))
        if len(out) >= n:
            break
    return out


def contractnli_questions(n: int, seed: int):
    """ContractNLI: legal-contract NLI (3-way: entailed/contradicted/not mentioned), long docs."""
    raw = load_dataset("presencesw/contract-nli", split="train").shuffle(seed=seed)
    lab2idx = {"entailment": 0, "notmentioned": 1, "contradiction": 2}
    opts = ["entailment", "not mentioned", "contradiction"]
    out = []
    for ex in raw:
        gold = lab2idx.get(str(ex.get("gold_label", "")).strip().lower())
        prem = str(ex.get("sentence1", "")).strip()
        hyp = str(ex.get("sentence2", "")).strip()
        if gold is None or not prem or not hyp:
            continue
        ctx = (f"Contract: {prem}\nStatement: {hyp}\n"
               "Q: Based on the contract, the statement is? A:")
        out.append((ctx, list(opts), gold))
        if len(out) >= n:
            break
    return out


def scinli_questions(n: int, seed: int):
    """SciNLI: scientific-paper NLI (4-way: entailment/contrasting/reasoning/neutral)."""
    raw = load_dataset("tasksource/scinli", split="train").shuffle(seed=seed)
    opts = ["entailment", "contrasting", "reasoning", "neutral"]
    lab2idx = {o: i for i, o in enumerate(opts)}
    out = []
    for ex in raw:
        gold = lab2idx.get(str(ex.get("label", "")).strip().lower())
        prem = str(ex.get("sentence1", "")).strip()
        hyp = str(ex.get("sentence2", "")).strip()
        if gold is None or not prem or not hyp:
            continue
        ctx = (f"Sentence 1: {prem}\nSentence 2: {hyp}\n"
               "Q: What is the relationship from sentence 1 to sentence 2? A:")
        out.append((ctx, list(opts), gold))
        if len(out) >= n:
            break
    return out


CANDIDATE_LOADERS = {
    "wanli": wanli_questions,
    "reclor": reclor_questions,
    "art": art_questions,
    "logiqa2": logiqa2_questions,
    "control": control_questions,
    "snli_hard": snli_hard_questions,
    "vitaminc": vitaminc_questions,
    "codah": codah_questions,
    "cosmosqa": cosmosqa_questions,
    "musr": musr_questions,
    "quail": quail_questions,
    "csqa": csqa_questions,
    "qasc": qasc_questions,
    "lingnli": lingnli_questions,
    "fevernli": fevernli_questions,
    "imppres": imppres_questions,
    "conjnli": conjnli_questions,
    "robustnli": robustnli_questions,
    "dynasent": dynasent_questions,
    "contractnli": contractnli_questions,
    "scinli": scinli_questions,
    "mmlupro": mmlupro_questions,
    "aquarat": aquarat_questions,
    "riddlesense": riddlesense_questions,
    "quality": quality_questions,
    "race": race_questions,
    "siqa": siqa_questions,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hellaswag",
                    choices=["hellaswag", "dream", "anli", "sciq", "wanli", "reclor", "art", "logiqa2",
                             "control", "snli_hard", "vitaminc", "codah", "cosmosqa", "musr", "quail",
                             "csqa", "qasc", "lingnli", "fevernli", "imppres", "conjnli", "robustnli", "dynasent", "quality", "race", "siqa", "mmlupro", "aquarat", "riddlesense", "contractnli", "scinli"])
    ap.add_argument("--anli-round", default="r2", choices=["r1", "r2", "r3"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--torch-dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--answer-suffix", default=DEFAULT_SUFFIX)
    args = ap.parse_args()

    if args.dataset == "hellaswag":
        qs = hellaswag_questions(args.n_questions, args.seed)
    elif args.dataset == "dream":
        qs = dream_questions(args.n_questions, args.seed)
    elif args.dataset == "anli":
        qs = anli_questions(args.n_questions, args.seed, args.anli_round)
    elif args.dataset == "sciq":
        qs = sciq_questions(args.n_questions, args.seed)
    else:  # candidate testbeds: wanli / reclor / art / logiqa2
        qs = CANDIDATE_LOADERS[args.dataset](args.n_questions, args.seed)
    print(f"dataset={args.dataset}  model={args.model}  questions={len(qs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=resolve_dtype(args.torch_dtype), low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(args.device)
    model.eval()

    native_correct = 0
    disc_correct = 0
    with torch.no_grad():
        for ctx, options, gold in qs:
            # NATIVE: length-normalised P(option | ctx)
            opt_scores = score_candidates(
                model, tokenizer, [ctx] * len(options), [" " + o for o in options], args.device, args.max_length
            ).tolist()
            native_norm = [
                s / max(1, len(tokenizer(" " + o, add_special_tokens=False)["input_ids"]))
                for o, s in zip(options, opt_scores)
            ]
            native_correct += int(int(np.argmax(native_norm)) == gold)

            # DISCRIMINATION: P(yes) for "{ctx} {option}{suffix}"
            disc_prompts = [f"{ctx} {o}{args.answer_suffix}" for o in options]
            yes = score_candidates(model, tokenizer, disc_prompts, [" yes"] * len(options), args.device, args.max_length)
            no = score_candidates(model, tokenizer, disc_prompts, [" no"] * len(options), args.device, args.max_length)
            p_yes = torch.softmax(torch.stack([no, yes], dim=1), dim=1)[:, 1]
            disc_correct += int(int(torch.argmax(p_yes)) == gold)

    n = len(qs)
    chance = 1.0 / len(qs[0][1]) if qs else float("nan")
    print(f"chance                                        : {chance:.3f}")
    print(f"NATIVE   (likelihood, length-normalised) acc  : {native_correct / n:.3f}")
    print(f"DISCRIM  (yes/no P(correct)) acc              : {disc_correct / n:.3f}")
    print("If NATIVE >> DISCRIM, the low discrimination accuracy is a yes/no-format limitation,")
    print("not a knowledge gap -- so the W2S lift partly reflects format learning / recall.")


if __name__ == "__main__":
    main()
