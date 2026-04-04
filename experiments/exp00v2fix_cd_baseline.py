#!/usr/bin/env python3
"""exp-00v2-fix: CD Baseline Revised Evaluation (Bug Fixes)
Fixes from exp-00v2:
1. TruthfulQA mc1 choices shuffled to remove position bias
2. TriviaQA / GSM8K use 5-shot prompting
3. Continuous CD metrics (cd_score, delta_logp) tracked per sample

Bug fixes over exp-00v2:
- REMOVED plausibility constraint from MC scoring (cd_score_choices).
  Plausibility constraint is a generation-time trick; for scoring existing
  completions, raw CD logits give correct log-probs.
- ADDED actual CD score computation for generative tasks (TriviaQA/GSM8K)
  by running both models on (prompt+response) post-generation.
- Plausibility constraint KEPT in greedy_decode_cd (correct usage).

Alpha range: {0.0, 0.5, 1.0, 1.5}  (alpha=0.0 = expert-only baseline)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import time
import logging
import gc
import re
import string
import random
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "/root/cd-circuit-mechanism/results"
ALPHAS = [0.0, 0.5, 1.0, 1.5]
NUM_SAMPLES = 200
MAX_GEN_TOKENS = 50
TOP_P = 0.9

EXPERT_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-9b"
AMATEUR_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-2b"

RNG = random.Random(42)


# ============================================================
# Utilities
# ============================================================

def print_gpu_info():
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu}, Memory: {mem:.1f} GB")
    else:
        logger.warning("No GPU available!")


def normalize_answer(s):
    """Normalize for exact-match: lowercase, strip articles/punctuation/whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s.strip()


def save_result(filename, data):
    """Save result JSON immediately."""
    path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved: {path}")


# ============================================================
# Dataset loading
# ============================================================

def load_all_datasets(n_samples):
    logger.info(f"Loading datasets (n={n_samples} samples each)...")
    datasets = {}

    logger.info("Loading TruthfulQA...")
    datasets["truthfulqa"] = load_dataset("truthful_qa", "multiple_choice", split=f"validation[:{n_samples}]")

    logger.info("Loading TriviaQA...")
    datasets["triviaqa"] = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{n_samples}]")
    datasets["triviaqa_train"] = load_dataset("trivia_qa", "rc.nocontext", split="train[:100]")

    logger.info("Loading GSM8K...")
    datasets["gsm8k"] = load_dataset("gsm8k", "main", split=f"test[:{n_samples}]")
    datasets["gsm8k_train"] = load_dataset("gsm8k", "main", split="train[:100]")

    logger.info("Loading HellaSwag...")
    datasets["hellaswag"] = load_dataset("Rowan/hellaswag", split=f"validation[:{n_samples}]")

    logger.info("All datasets loaded.")
    return datasets


# ============================================================
# TruthfulQA shuffle
# ============================================================

def shuffle_mc1_choices(item):
    """Shuffle mc1 choices and labels together, return (choices, labels, correct_idx)."""
    choices = list(item['mc1_targets']['choices'])
    labels = list(item['mc1_targets']['labels'])
    pairs = list(zip(choices, labels))
    RNG.shuffle(pairs)
    shuffled_choices = [p[0] for p in pairs]
    shuffled_labels = [p[1] for p in pairs]
    correct_idx = shuffled_labels.index(1)
    return shuffled_choices, shuffled_labels, correct_idx


# ============================================================
# Few-shot prompt builders
# ============================================================

def build_triviaqa_fewshot_prefix(train_data):
    """Build 5-shot prefix from training data."""
    indices = list(range(len(train_data)))
    RNG.shuffle(indices)
    selected = indices[:5]
    parts = []
    for i in selected:
        item = train_data[i]
        q = item['question']
        a = item['answer']['value']
        parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts) + "\n\n"


def build_gsm8k_fewshot_prefix(train_data):
    """Build 5-shot CoT prefix from training data."""
    indices = list(range(len(train_data)))
    RNG.shuffle(indices)
    selected = indices[:5]
    parts = []
    for i in selected:
        item = train_data[i]
        q = item['question']
        answer_text = item['answer']
        parts.append(f"Q: {q}\nA: Let's think step by step. {answer_text}")
    return "\n\n".join(parts) + "\n\n"


# ============================================================
# Core scoring functions
# ============================================================

def get_completion_logits(model, tokenizer, prompt, completion, device):
    """Get logits at each completion token position. Returns (logits_tensor, token_ids)."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    completion_ids = full_ids[len(prompt_ids):]

    if len(completion_ids) == 0:
        return None, None

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # [seq_len, vocab]

    positions = [len(prompt_ids) + i - 1 for i in range(len(completion_ids))]
    positions = [p for p in positions if 0 <= p < logits.shape[0]]

    if not positions:
        return None, None

    return logits[positions, :], completion_ids[:len(positions)]


def cd_score_choices(expert_logits_list, amateur_logits_list, comp_ids_list, alpha):
    """Compute CD log-likelihood for multiple choices (MC scoring).

    NO plausibility constraint here — that is a generation-time trick.
    For scoring existing completions, raw CD logits give correct log-probs.

    Returns list of (cd_score, expert_logp, cd_logp) tuples.
    """
    results = []
    for e_logits, a_logits, comp_ids in zip(expert_logits_list, amateur_logits_list, comp_ids_list):
        if e_logits is None or a_logits is None:
            results.append((float("-inf"), float("-inf"), float("-inf")))
            continue

        # Expert log prob (no CD)
        expert_lp = torch.log_softmax(e_logits, dim=-1)
        expert_score = sum(expert_lp[i, comp_ids[i]].item() for i in range(min(len(comp_ids), expert_lp.shape[0])))
        expert_score /= max(len(comp_ids), 1)

        if alpha == 0.0:
            results.append((expert_score, expert_score, expert_score))
            continue

        # CD logits — NO plausibility mask for scoring
        cd_logits = e_logits - alpha * a_logits
        cd_lp = torch.log_softmax(cd_logits, dim=-1)
        cd_score = sum(cd_lp[i, comp_ids[i]].item() for i in range(min(len(comp_ids), cd_lp.shape[0])))
        cd_score /= max(len(comp_ids), 1)

        results.append((cd_score, expert_score, cd_score))

    return results


def compute_generation_cd_scores(expert_model, amateur_model, tokenizer, prompt, response, alpha, device):
    """Compute per-token CD log-probs for a generated response.

    Runs both models on (prompt + response) and computes CD scores at each
    response token position. No plausibility constraint (scoring, not generation).

    Returns (cd_score_mean, expert_logp_mean, delta_logp).
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_text = prompt + response
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    response_ids = full_ids[len(prompt_ids):]

    if len(response_ids) == 0:
        return 0.0, 0.0, 0.0

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        e_logits = expert_model(input_ids).logits[0]  # [seq_len, vocab]

    # Positions: for response token i, the predicting position is (prompt_len + i - 1)
    positions = [len(prompt_ids) + i - 1 for i in range(len(response_ids))]
    positions = [p for p in positions if 0 <= p < e_logits.shape[0]]
    n_pos = len(positions)

    if n_pos == 0:
        return 0.0, 0.0, 0.0

    # Expert log prob
    expert_lp = torch.log_softmax(e_logits[positions], dim=-1)
    expert_score = sum(expert_lp[i, response_ids[i]].item() for i in range(min(len(response_ids), n_pos)))
    expert_score /= n_pos

    if alpha == 0.0:
        return expert_score, expert_score, 0.0

    # Amateur forward pass
    with torch.no_grad():
        a_logits = amateur_model(input_ids).logits[0]

    # CD log prob
    cd_logits = e_logits[positions] - alpha * a_logits[positions]
    cd_lp = torch.log_softmax(cd_logits, dim=-1)
    cd_score = sum(cd_lp[i, response_ids[i]].item() for i in range(min(len(response_ids), n_pos)))
    cd_score /= n_pos

    delta = cd_score - expert_score
    return cd_score, expert_score, delta


def greedy_decode(model, tokenizer, prompt, device, max_tokens=MAX_GEN_TOKENS):
    """Standard greedy decode."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()

    for _ in range(max_tokens):
        inp = torch.tensor([generated], device=device)
        with torch.no_grad():
            logits = model(inp).logits[0, -1, :]
        next_token = logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return tokenizer.decode(generated[len(input_ids[0]):], skip_special_tokens=True)


def greedy_decode_cd(expert_model, amateur_model, tokenizer, prompt, alpha, device, max_tokens=MAX_GEN_TOKENS):
    """Greedy decode using CD logits with plausibility constraint.
    Plausibility constraint is correct here — it filters generation candidates.
    alpha=0 falls back to expert-only.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(input_ids)

    for _ in range(max_tokens):
        inp = torch.tensor([generated], device=device)
        with torch.no_grad():
            e_logits = expert_model(inp).logits[0, -1, :]
            if alpha > 0:
                a_logits = amateur_model(inp).logits[0, -1, :]
                cd_logits = e_logits - alpha * a_logits
                # Plausibility constraint (correct for generation)
                expert_probs = torch.softmax(e_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(expert_probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > TOP_P
                cd_logits[sorted_indices[mask]] = float("-inf")
            else:
                cd_logits = e_logits

        next_token = cd_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)


# ============================================================
# Evaluation functions
# ============================================================

def eval_truthfulqa(expert_model, amateur_model, tokenizer, dataset, alpha, device):
    """Evaluate TruthfulQA mc1 with shuffled choices."""
    per_sample = []
    correct_count = 0

    for idx, item in enumerate(dataset):
        choices, labels, correct_idx = shuffle_mc1_choices(item)
        prompt = f"Q: {item['question']}\nA:"

        e_logits_list, e_ids_list = [], []
        a_logits_list, a_ids_list = [], []
        for choice in choices:
            e_log, e_ids = get_completion_logits(expert_model, tokenizer, prompt, f" {choice}", device)
            e_logits_list.append(e_log)
            e_ids_list.append(e_ids)

            if alpha > 0:
                a_log, a_ids = get_completion_logits(amateur_model, tokenizer, prompt, f" {choice}", device)
            else:
                a_log, a_ids = e_log, e_ids
            a_logits_list.append(a_log)
            a_ids_list.append(a_ids)

        score_tuples = cd_score_choices(e_logits_list, a_logits_list, e_ids_list, alpha)
        cd_scores = [t[0] for t in score_tuples]
        expert_logps = [t[1] for t in score_tuples]
        cd_logps = [t[2] for t in score_tuples]

        pred = int(np.argmax(cd_scores))
        is_correct = pred == correct_idx

        if is_correct:
            correct_count += 1

        per_sample.append({
            "idx": idx,
            "correct": is_correct,
            "cd_score_correct": cd_scores[correct_idx],
            "cd_score_top": cd_scores[pred],
            "expert_logp_correct": expert_logps[correct_idx],
            "cd_logp_correct": cd_logps[correct_idx],
            "delta_logp": cd_logps[correct_idx] - expert_logps[correct_idx],
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  TruthfulQA [{idx+1}/{len(dataset)}] acc={correct_count/(idx+1):.3f}")

    accuracy = correct_count / len(dataset)
    cd_scores_correct = [s["cd_score_correct"] for s in per_sample]
    delta_logps = [s["delta_logp"] for s in per_sample]

    return {
        "dataset": "truthfulqa_shuffled",
        "alpha": alpha,
        "n_samples": len(dataset),
        "metrics": {
            "accuracy": accuracy,
            "cd_score_mean": float(np.mean(cd_scores_correct)),
            "cd_score_std": float(np.std(cd_scores_correct)),
            "delta_logp_mean": float(np.mean(delta_logps)),
            "delta_logp_std": float(np.std(delta_logps)),
        },
        "per_sample": per_sample,
        "timestamp": datetime.now().isoformat(),
    }


def eval_hellaswag(expert_model, amateur_model, tokenizer, dataset, alpha, device):
    """Evaluate HellaSwag (multiple choice)."""
    per_sample = []
    correct_count = 0

    for idx, item in enumerate(dataset):
        prompt = item["ctx"]
        choices = item["endings"]
        correct_idx = int(item["label"])

        e_logits_list, e_ids_list = [], []
        a_logits_list, a_ids_list = [], []
        for choice in choices:
            e_log, e_ids = get_completion_logits(expert_model, tokenizer, prompt, f" {choice}", device)
            e_logits_list.append(e_log)
            e_ids_list.append(e_ids)

            if alpha > 0:
                a_log, a_ids = get_completion_logits(amateur_model, tokenizer, prompt, f" {choice}", device)
            else:
                a_log, a_ids = e_log, e_ids
            a_logits_list.append(a_log)
            a_ids_list.append(a_ids)

        score_tuples = cd_score_choices(e_logits_list, a_logits_list, e_ids_list, alpha)
        cd_scores = [t[0] for t in score_tuples]
        expert_logps = [t[1] for t in score_tuples]
        cd_logps = [t[2] for t in score_tuples]

        pred = int(np.argmax(cd_scores))
        is_correct = pred == correct_idx

        if is_correct:
            correct_count += 1

        per_sample.append({
            "idx": idx,
            "correct": is_correct,
            "cd_score_correct": cd_scores[correct_idx],
            "cd_score_top": cd_scores[pred],
            "expert_logp_correct": expert_logps[correct_idx],
            "cd_logp_correct": cd_logps[correct_idx],
            "delta_logp": cd_logps[correct_idx] - expert_logps[correct_idx],
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  HellaSwag [{idx+1}/{len(dataset)}] acc={correct_count/(idx+1):.3f}")

    accuracy = correct_count / len(dataset)
    cd_scores_correct = [s["cd_score_correct"] for s in per_sample]
    delta_logps = [s["delta_logp"] for s in per_sample]

    return {
        "dataset": "hellaswag",
        "alpha": alpha,
        "n_samples": len(dataset),
        "metrics": {
            "accuracy": accuracy,
            "cd_score_mean": float(np.mean(cd_scores_correct)),
            "cd_score_std": float(np.std(cd_scores_correct)),
            "delta_logp_mean": float(np.mean(delta_logps)),
            "delta_logp_std": float(np.std(delta_logps)),
        },
        "per_sample": per_sample,
        "timestamp": datetime.now().isoformat(),
    }


def eval_triviaqa(expert_model, amateur_model, tokenizer, dataset, train_data, alpha, device):
    """Evaluate TriviaQA with 5-shot prompting + actual CD score computation."""
    fewshot_prefix = build_triviaqa_fewshot_prefix(train_data)
    per_sample = []
    correct_count = 0

    for idx, item in enumerate(dataset):
        prompt = fewshot_prefix + f"Q: {item['question']}\nA:"
        response = greedy_decode_cd(expert_model, amateur_model, tokenizer, prompt, alpha, device)
        response_first_line = response.split("\n")[0].strip()

        # Exact match with normalization
        pred_normalized = normalize_answer(response_first_line)
        aliases = item["answer"]["aliases"] + [item["answer"]["value"]]
        is_correct = any(normalize_answer(a) == pred_normalized for a in aliases)
        if not is_correct:
            is_correct = any(normalize_answer(a) in pred_normalized for a in aliases)

        if is_correct:
            correct_count += 1

        # Compute actual CD scores for the generated response
        cd_score, expert_logp, delta_logp = compute_generation_cd_scores(
            expert_model, amateur_model, tokenizer, prompt, response, alpha, device
        )

        per_sample.append({
            "idx": idx,
            "correct": is_correct,
            "response": response.strip()[:200],
            "cd_score_correct": cd_score,
            "cd_score_top": cd_score,
            "expert_logp_correct": expert_logp,
            "cd_logp_correct": cd_score,
            "delta_logp": delta_logp,
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  TriviaQA [{idx+1}/{len(dataset)}] acc={correct_count/(idx+1):.3f}")

    accuracy = correct_count / len(dataset)
    cd_scores_all = [s["cd_score_correct"] for s in per_sample]
    delta_logps = [s["delta_logp"] for s in per_sample]

    return {
        "dataset": "triviaqa_5shot",
        "alpha": alpha,
        "n_samples": len(dataset),
        "metrics": {
            "accuracy": accuracy,
            "cd_score_mean": float(np.mean(cd_scores_all)),
            "cd_score_std": float(np.std(cd_scores_all)),
            "delta_logp_mean": float(np.mean(delta_logps)),
            "delta_logp_std": float(np.std(delta_logps)),
        },
        "per_sample": per_sample,
        "timestamp": datetime.now().isoformat(),
    }


def eval_gsm8k(expert_model, amateur_model, tokenizer, dataset, train_data, alpha, device):
    """Evaluate GSM8K with 5-shot CoT prompting + actual CD score computation."""
    fewshot_prefix = build_gsm8k_fewshot_prefix(train_data)
    per_sample = []
    correct_count = 0

    for idx, item in enumerate(dataset):
        prompt = fewshot_prefix + f"Q: {item['question']}\nA: Let's think step by step."
        response = greedy_decode_cd(expert_model, amateur_model, tokenizer, prompt, alpha, device,
                                     max_tokens=150)

        # Extract number after ####
        gold_answer = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else ""
        pred_answer = ""
        if "####" in response:
            pred_answer = response.split("####")[-1].strip().split()[0] if response.split("####")[-1].strip() else ""
        else:
            numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', response)
            if numbers:
                pred_answer = numbers[-1].replace(",", "")

        gold_clean = gold_answer.replace(",", "").strip()
        pred_clean = pred_answer.replace(",", "").strip()
        is_correct = gold_clean != "" and gold_clean == pred_clean

        if is_correct:
            correct_count += 1

        # Compute actual CD scores for the generated response
        cd_score, expert_logp, delta_logp = compute_generation_cd_scores(
            expert_model, amateur_model, tokenizer, prompt, response, alpha, device
        )

        per_sample.append({
            "idx": idx,
            "correct": is_correct,
            "response": response.strip()[:300],
            "gold": gold_clean,
            "pred": pred_clean,
            "cd_score_correct": cd_score,
            "cd_score_top": cd_score,
            "expert_logp_correct": expert_logp,
            "cd_logp_correct": cd_score,
            "delta_logp": delta_logp,
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  GSM8K [{idx+1}/{len(dataset)}] acc={correct_count/(idx+1):.3f}")

    accuracy = correct_count / len(dataset)
    cd_scores_all = [s["cd_score_correct"] for s in per_sample]
    delta_logps = [s["delta_logp"] for s in per_sample]

    return {
        "dataset": "gsm8k_5shot_cot",
        "alpha": alpha,
        "n_samples": len(dataset),
        "metrics": {
            "accuracy": accuracy,
            "cd_score_mean": float(np.mean(cd_scores_all)),
            "cd_score_std": float(np.std(cd_scores_all)),
            "delta_logp_mean": float(np.mean(delta_logps)),
            "delta_logp_std": float(np.std(delta_logps)),
        },
        "per_sample": per_sample,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_gpu_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples = 5 if args.dry_run else NUM_SAMPLES
    if args.dry_run:
        logger.info("*** DRY RUN MODE ***")

    datasets = load_all_datasets(n_samples)

    # ==============================
    # PHASE 0: TruthfulQA shuffle verification
    # ==============================
    logger.info("\n" + "="*60)
    logger.info("PHASE 0: TruthfulQA shuffle verification (alpha=0.5)")
    logger.info("="*60)

    logger.info("Loading expert model...")
    expert_model = AutoModelForCausalLM.from_pretrained(
        EXPERT_LOCAL, torch_dtype=torch.float16, device_map="cuda"
    )
    expert_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(EXPERT_LOCAL)
    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Expert loaded. GPU mem: {mem:.1f} GB")

    logger.info("Loading amateur model...")
    amateur_model = AutoModelForCausalLM.from_pretrained(
        AMATEUR_LOCAL, torch_dtype=torch.float16, device_map="cuda"
    )
    amateur_model.eval()
    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Both models loaded. GPU mem: {mem:.1f} GB")

    # Run shuffle verification
    verify_result = eval_truthfulqa(expert_model, amateur_model, tokenizer,
                                     datasets["truthfulqa"], alpha=0.5, device=device)
    save_result("exp00v2fix_truthfulqa_shuffle_verify.json", verify_result)

    # Sanity check: CD scores should NOT be -inf
    n_inf = sum(1 for s in verify_result["per_sample"] if s["cd_logp_correct"] == float("-inf"))
    logger.info(f"SHUFFLE VERIFY: accuracy={verify_result['metrics']['accuracy']:.3f}, "
                f"cd_score_mean={verify_result['metrics']['cd_score_mean']:.4f}, "
                f"n_inf_scores={n_inf}/{len(verify_result['per_sample'])}")
    if n_inf > 0:
        logger.error(f"WARNING: {n_inf} samples have -inf CD scores! Bug may not be fully fixed.")

    # ==============================
    # PHASE 1-4: Full evaluation across all alphas
    # ==============================
    for alpha in ALPHAS:
        logger.info(f"\n{'='*60}")
        logger.info(f"FULL EVAL: alpha={alpha}")
        logger.info(f"{'='*60}")

        # TruthfulQA (shuffled)
        logger.info(f"[alpha={alpha}] TruthfulQA (shuffled)...")
        tqa_result = eval_truthfulqa(expert_model, amateur_model, tokenizer,
                                      datasets["truthfulqa"], alpha, device)
        save_result(f"exp00v2fix_truthfulqa_alpha{alpha}.json", tqa_result)
        logger.info(f"  -> accuracy={tqa_result['metrics']['accuracy']:.3f}, "
                     f"cd_score_mean={tqa_result['metrics']['cd_score_mean']:.4f}")

        # HellaSwag
        logger.info(f"[alpha={alpha}] HellaSwag...")
        hs_result = eval_hellaswag(expert_model, amateur_model, tokenizer,
                                    datasets["hellaswag"], alpha, device)
        save_result(f"exp00v2fix_hellaswag_alpha{alpha}.json", hs_result)
        logger.info(f"  -> accuracy={hs_result['metrics']['accuracy']:.3f}, "
                     f"cd_score_mean={hs_result['metrics']['cd_score_mean']:.4f}")

        # TriviaQA (5-shot)
        logger.info(f"[alpha={alpha}] TriviaQA (5-shot)...")
        tqa_open_result = eval_triviaqa(expert_model, amateur_model, tokenizer,
                                         datasets["triviaqa"], datasets["triviaqa_train"],
                                         alpha, device)
        save_result(f"exp00v2fix_triviaqa_alpha{alpha}.json", tqa_open_result)
        logger.info(f"  -> accuracy={tqa_open_result['metrics']['accuracy']:.3f}, "
                     f"cd_score_mean={tqa_open_result['metrics']['cd_score_mean']:.4f}")

        # GSM8K (5-shot CoT)
        logger.info(f"[alpha={alpha}] GSM8K (5-shot CoT)...")
        gsm_result = eval_gsm8k(expert_model, amateur_model, tokenizer,
                                 datasets["gsm8k"], datasets["gsm8k_train"],
                                 alpha, device)
        save_result(f"exp00v2fix_gsm8k_alpha{alpha}.json", gsm_result)
        logger.info(f"  -> accuracy={gsm_result['metrics']['accuracy']:.3f}, "
                     f"cd_score_mean={gsm_result['metrics']['cd_score_mean']:.4f}")

    # Cleanup
    del expert_model, amateur_model
    torch.cuda.empty_cache()
    gc.collect()

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("EXP-00v2-FIX COMPLETE")
    logger.info("="*60)
    logger.info(f"{'Alpha':<8} {'TruthfulQA':>12} {'HellaSwag':>12} {'TriviaQA':>12} {'GSM8K':>12}")
    logger.info("-" * 58)
    for alpha in ALPHAS:
        accs = []
        for ds in ["truthfulqa", "hellaswag", "triviaqa", "gsm8k"]:
            fname = f"exp00v2fix_{ds}_alpha{alpha}.json"
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                accs.append(data["metrics"]["accuracy"])
            else:
                accs.append(float("nan"))
        logger.info(f"{alpha:<8.1f} {accs[0]:>12.3f} {accs[1]:>12.3f} {accs[2]:>12.3f} {accs[3]:>12.3f}")


if __name__ == "__main__":
    main()
