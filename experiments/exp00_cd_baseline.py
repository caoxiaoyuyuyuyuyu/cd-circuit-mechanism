#!/usr/bin/env python3
"""exp-00: CD Baseline Performance Test
Hypothesis: CD gains larger on knowledge-intensive tasks (TruthfulQA/TriviaQA)
than reasoning tasks (GSM8K/HellaSwag).

Uses plain transformers (no TransformerLens needed for baseline logit comparison).
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import time
import logging
import gc
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "/root/cd-circuit-mechanism/results/exp00_results.json"
ALPHAS = [0.5, 1.0]
NUM_SAMPLES = 200
MAX_GEN_TOKENS = 50
TOP_P = 0.9

EXPERT_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-9b"
AMATEUR_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-2b"


def print_gpu_info():
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu}, Memory: {mem:.1f} GB")
    else:
        logger.warning("No GPU available!")


def load_datasets(dry_run=False):
    n = 3 if dry_run else NUM_SAMPLES
    logger.info(f"Loading datasets (n={n} samples each)...")

    datasets = {}

    logger.info("Loading TruthfulQA...")
    datasets["TruthfulQA"] = load_dataset("truthful_qa", "multiple_choice", split=f"validation[:{n}]")

    logger.info("Loading TriviaQA...")
    datasets["TriviaQA"] = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{n}]")

    logger.info("Loading GSM8K...")
    datasets["GSM8K"] = load_dataset("gsm8k", "main", split=f"test[:{n}]")

    logger.info("Loading HellaSwag...")
    datasets["HellaSwag"] = load_dataset("Rowan/hellaswag", split=f"validation[:{n}]")

    logger.info("All datasets loaded.")
    return datasets


def get_avg_log_prob(model, tokenizer, prompt, completion, device):
    """Get average log prob of completion tokens given prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    completion_len = len(full_ids) - len(prompt_ids)

    if completion_len <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, seq_len, vocab]

    log_probs = torch.log_softmax(logits[0], dim=-1)

    total = 0.0
    for i in range(completion_len):
        pos = len(prompt_ids) + i - 1  # predict next token
        token_id = full_ids[len(prompt_ids) + i]
        if pos >= 0:
            total += log_probs[pos, token_id].item()

    return total / completion_len


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
    """Compute CD log-likelihood for multiple choices with plausibility constraint."""
    scores = []
    for e_logits, a_logits, comp_ids in zip(expert_logits_list, amateur_logits_list, comp_ids_list):
        if e_logits is None or a_logits is None:
            scores.append(float("-inf"))
            continue

        cd_logits = e_logits - alpha * a_logits

        # Plausibility constraint: top-p on expert
        expert_probs = torch.softmax(e_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(expert_probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > TOP_P
        for pos in range(cd_logits.shape[0]):
            cd_logits[pos, sorted_indices[pos][mask[pos]]] = float("-inf")

        cd_lp = torch.log_softmax(cd_logits, dim=-1)
        total = sum(cd_lp[i, comp_ids[i]].item() for i in range(min(len(comp_ids), cd_lp.shape[0])))
        scores.append(total / max(len(comp_ids), 1))

    return scores


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
    """Greedy decode using CD logits."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(input_ids)

    for _ in range(max_tokens):
        inp = torch.tensor([generated], device=device)
        with torch.no_grad():
            e_logits = expert_model(inp).logits[0, -1, :]
            a_logits = amateur_model(inp).logits[0, -1, :]

        cd_logits = e_logits - alpha * a_logits

        # Plausibility constraint
        expert_probs = torch.softmax(e_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(expert_probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > TOP_P
        cd_logits[sorted_indices[mask]] = float("-inf")

        next_token = cd_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)


# ---- Phase 1: Expert-only evaluation ----

def eval_expert_mc(model, tokenizer, dataset, ds_name, prompt_fn, choices_fn, label_fn, device):
    """Evaluate expert on multiple-choice dataset. Returns list of dicts with logits."""
    results = []
    for idx, item in enumerate(dataset):
        prompt = prompt_fn(item)
        choices = choices_fn(item)
        label = label_fn(item)

        choice_logits = []
        choice_ids = []
        expert_log_probs = []
        for choice in choices:
            logits, cids = get_completion_logits(model, tokenizer, prompt, f" {choice}", device)
            choice_logits.append(logits.cpu() if logits is not None else None)
            choice_ids.append(cids)
            if logits is not None:
                lp = torch.log_softmax(logits, dim=-1)
                avg = sum(lp[i, cids[i]].item() for i in range(min(len(cids), logits.shape[0]))) / max(len(cids), 1)
                expert_log_probs.append(avg)
            else:
                expert_log_probs.append(float("-inf"))

        pred = int(np.argmax(expert_log_probs))
        results.append({
            "correct_idx": label,
            "expert_pred": pred,
            "expert_correct": pred == label,
            "choice_logits": choice_logits,
            "choice_ids": choice_ids,
        })

        if (idx + 1) % 20 == 0:
            acc = sum(r["expert_correct"] for r in results) / len(results)
            logger.info(f"  {ds_name} [{idx+1}/{len(dataset)}] expert_acc={acc:.3f}")

    return results


def eval_expert_open(model, tokenizer, dataset, ds_name, prompt_fn, answer_check_fn, device):
    """Evaluate expert on open-ended dataset via greedy decode."""
    results = []
    for idx, item in enumerate(dataset):
        prompt = prompt_fn(item)
        response = greedy_decode(model, tokenizer, prompt, device)
        correct = answer_check_fn(item, response)

        results.append({
            "expert_response": response.strip(),
            "expert_correct": correct,
        })

        if (idx + 1) % 20 == 0:
            acc = sum(r["expert_correct"] for r in results) / len(results)
            logger.info(f"  {ds_name} [{idx+1}/{len(dataset)}] expert_acc={acc:.3f}")

    return results


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_gpu_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        logger.info("*** DRY RUN MODE (3 samples per dataset) ***")

    datasets = load_datasets(dry_run=args.dry_run)

    # ==============================
    # PHASE 1: Expert model
    # ==============================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Loading expert model (gemma-2-9b)")
    logger.info("="*60)

    expert_model = AutoModelForCausalLM.from_pretrained(
        EXPERT_LOCAL, torch_dtype=torch.float16, device_map="cuda"
    )
    expert_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(EXPERT_LOCAL)

    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Expert loaded. GPU mem: {mem:.1f} GB")

    # --- Prompt/label functions ---
    def tqa_prompt(item): return f"Q: {item['question']}\nA:"
    def tqa_choices(item): return item["mc1_targets"]["choices"]
    def tqa_label(item): return item["mc1_targets"]["labels"].index(1)

    def hs_prompt(item): return item["ctx"]
    def hs_choices(item): return item["endings"]
    def hs_label(item): return int(item["label"])

    def tqa_open_prompt(item): return f"Q: {item['question']}\nA:"
    def tqa_open_check(item, resp):
        return any(a.lower() in resp.strip().lower() for a in item["answer"]["aliases"])

    def gsm_prompt(item): return f"Q: {item['question']}\nA: Let's solve step by step."
    def gsm_check(item, resp):
        fa = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else ""
        return fa != "" and fa in resp

    logger.info("Expert: TruthfulQA...")
    expert_tqa = eval_expert_mc(expert_model, tokenizer, datasets["TruthfulQA"], "TruthfulQA",
                                 tqa_prompt, tqa_choices, tqa_label, device)

    logger.info("Expert: HellaSwag...")
    expert_hs = eval_expert_mc(expert_model, tokenizer, datasets["HellaSwag"], "HellaSwag",
                                hs_prompt, hs_choices, hs_label, device)

    logger.info("Expert: TriviaQA...")
    expert_trivia = eval_expert_open(expert_model, tokenizer, datasets["TriviaQA"], "TriviaQA",
                                      tqa_open_prompt, tqa_open_check, device)

    logger.info("Expert: GSM8K...")
    expert_gsm = eval_expert_open(expert_model, tokenizer, datasets["GSM8K"], "GSM8K",
                                   gsm_prompt, gsm_check, device)

    # ==============================
    # PHASE 2: Load amateur, keep expert for CD decode
    # ==============================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Loading amateur model (gemma-2-2b)")
    logger.info("="*60)

    amateur_model = AutoModelForCausalLM.from_pretrained(
        AMATEUR_LOCAL, torch_dtype=torch.float16, device_map="cuda"
    )
    amateur_model.eval()
    amateur_tokenizer = AutoTokenizer.from_pretrained(AMATEUR_LOCAL)

    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Both models loaded. GPU mem: {mem:.1f} GB")

    # ==============================
    # PHASE 3: CD evaluation
    # ==============================
    final_results = {}

    for alpha in ALPHAS:
        logger.info(f"\n{'='*60}")
        logger.info(f"CD evaluation with alpha={alpha}")
        logger.info(f"{'='*60}")
        alpha_results = {}

        # --- TruthfulQA CD ---
        logger.info(f"CD alpha={alpha}: TruthfulQA...")
        cd_correct = 0
        for idx, (item, ed) in enumerate(zip(datasets["TruthfulQA"], expert_tqa)):
            prompt = tqa_prompt(item)
            choices = tqa_choices(item)

            a_logits_list, a_ids_list = [], []
            for choice in choices:
                logits, cids = get_completion_logits(amateur_model, amateur_tokenizer, prompt, f" {choice}", device)
                a_logits_list.append(logits)
                a_ids_list.append(cids)

            e_logits_list = [(l.to(device) if l is not None else None) for l in ed["choice_logits"]]
            scores = cd_score_choices(e_logits_list, a_logits_list, ed["choice_ids"], alpha)

            if int(np.argmax(scores)) == ed["correct_idx"]:
                cd_correct += 1

            if (idx + 1) % 20 == 0:
                logger.info(f"  TruthfulQA CD [{idx+1}] acc={cd_correct/(idx+1):.3f}")

        e_acc = sum(d["expert_correct"] for d in expert_tqa) / len(expert_tqa)
        c_acc = cd_correct / len(expert_tqa)
        alpha_results["TruthfulQA"] = {"accuracy_expert": e_acc, "accuracy_cd": c_acc, "cd_gain": c_acc - e_acc}
        logger.info(f"TruthfulQA: expert={e_acc:.3f}, cd={c_acc:.3f}, gain={c_acc-e_acc:+.3f}")

        # --- HellaSwag CD ---
        logger.info(f"CD alpha={alpha}: HellaSwag...")
        cd_correct = 0
        for idx, (item, ed) in enumerate(zip(datasets["HellaSwag"], expert_hs)):
            prompt = hs_prompt(item)
            endings = hs_choices(item)

            a_logits_list, a_ids_list = [], []
            for ending in endings:
                logits, cids = get_completion_logits(amateur_model, amateur_tokenizer, prompt, f" {ending}", device)
                a_logits_list.append(logits)
                a_ids_list.append(cids)

            e_logits_list = [(l.to(device) if l is not None else None) for l in ed["choice_logits"]]
            scores = cd_score_choices(e_logits_list, a_logits_list, ed["choice_ids"], alpha)

            if int(np.argmax(scores)) == ed["correct_idx"]:
                cd_correct += 1

            if (idx + 1) % 20 == 0:
                logger.info(f"  HellaSwag CD [{idx+1}] acc={cd_correct/(idx+1):.3f}")

        e_acc = sum(d["expert_correct"] for d in expert_hs) / len(expert_hs)
        c_acc = cd_correct / len(expert_hs)
        alpha_results["HellaSwag"] = {"accuracy_expert": e_acc, "accuracy_cd": c_acc, "cd_gain": c_acc - e_acc}
        logger.info(f"HellaSwag: expert={e_acc:.3f}, cd={c_acc:.3f}, gain={c_acc-e_acc:+.3f}")

        # --- TriviaQA CD decode ---
        logger.info(f"CD alpha={alpha}: TriviaQA (CD decode)...")
        cd_correct = 0
        for idx, (item, ed) in enumerate(zip(datasets["TriviaQA"], expert_trivia)):
            prompt = tqa_open_prompt(item)
            response = greedy_decode_cd(expert_model, amateur_model, tokenizer, prompt, alpha, device)
            if tqa_open_check(item, response):
                cd_correct += 1

            if (idx + 1) % 20 == 0:
                logger.info(f"  TriviaQA CD [{idx+1}] acc={cd_correct/(idx+1):.3f}")

        e_acc = sum(d["expert_correct"] for d in expert_trivia) / len(expert_trivia)
        c_acc = cd_correct / len(expert_trivia)
        alpha_results["TriviaQA"] = {"accuracy_expert": e_acc, "accuracy_cd": c_acc, "cd_gain": c_acc - e_acc}
        logger.info(f"TriviaQA: expert={e_acc:.3f}, cd={c_acc:.3f}, gain={c_acc-e_acc:+.3f}")

        # --- GSM8K CD decode ---
        logger.info(f"CD alpha={alpha}: GSM8K (CD decode)...")
        cd_correct = 0
        for idx, (item, ed) in enumerate(zip(datasets["GSM8K"], expert_gsm)):
            prompt = gsm_prompt(item)
            response = greedy_decode_cd(expert_model, amateur_model, tokenizer, prompt, alpha, device)
            if gsm_check(item, response):
                cd_correct += 1

            if (idx + 1) % 20 == 0:
                logger.info(f"  GSM8K CD [{idx+1}] acc={cd_correct/(idx+1):.3f}")

        e_acc = sum(d["expert_correct"] for d in expert_gsm) / len(expert_gsm)
        c_acc = cd_correct / len(expert_gsm)
        alpha_results["GSM8K"] = {"accuracy_expert": e_acc, "accuracy_cd": c_acc, "cd_gain": c_acc - e_acc}
        logger.info(f"GSM8K: expert={e_acc:.3f}, cd={c_acc:.3f}, gain={c_acc-e_acc:+.3f}")

        final_results[f"alpha_{alpha}"] = alpha_results

    # Cleanup
    del expert_model, amateur_model
    torch.cuda.empty_cache()
    gc.collect()

    # Save results
    output = {
        "experiment": "exp-00",
        "description": "CD Baseline Performance Test",
        "hypothesis": "Knowledge-intensive tasks show larger CD gains than reasoning tasks",
        "expert_model": "google/gemma-2-9b",
        "amateur_model": "google/gemma-2-2b",
        "alphas": ALPHAS,
        "num_samples": 3 if args.dry_run else NUM_SAMPLES,
        "dry_run": args.dry_run,
        "timestamp": datetime.now().isoformat(),
        "results": final_results,
    }

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*60)
    for alpha_key, alpha_res in final_results.items():
        logger.info(f"\n--- {alpha_key} ---")
        logger.info(f"{'Dataset':<15} {'Expert':>8} {'CD':>8} {'Gain':>8}")
        logger.info("-" * 41)
        for ds_name, metrics in alpha_res.items():
            logger.info(
                f"{ds_name:<15} {metrics['accuracy_expert']:>8.3f} "
                f"{metrics['accuracy_cd']:>8.3f} {metrics['cd_gain']:>+8.3f}"
            )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
