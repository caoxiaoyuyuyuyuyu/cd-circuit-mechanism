#!/usr/bin/env python3
"""Exp-03: Per-Layer Logit Diff Anatomy for Contrastive Decoding

Analyzes per-layer logit contributions from expert (Gemma-2-9B) and amateur
(Gemma-2-2B) to understand where CD gain originates in the residual stream.

Method:
  For each prompt, hook every layer's residual stream (resid_post), project it
  through the model's unembedding head, and compute:
    - Per-layer logit at target token position
    - Layer-wise delta (contribution of each layer)
    - CD contribution = expert_delta[l] - alpha * amateur_delta[l]

  Samples are split into CD-effective (CD improves over expert) and
  CD-ineffective groups for comparative analysis.

Primary analysis: per-model layer contributions (no cross-model alignment).
CD score computed in logit space at final layer only.

Datasets: HellaSwag (MC), TruthfulQA (MC) from exp-00v2-fix results.

NOTE on prompt format: This script uses MC format "(A) choice1 (B) choice2..."
while exp-00v2-fix scores each choice independently (per-choice logprob).
This is a design choice — we analyze per-layer logit toward the correct
answer token, not the MC selection mechanism. TruthfulQA shuffle uses
per-sample deterministic seeds (42+idx), not exp-00v2-fix's shared RNG.
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
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

EXPERT_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-9b"
EXPERT_HF = "google/gemma-2-9b"
AMATEUR_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-2b"
AMATEUR_HF = "google/gemma-2-2b"
RESULTS_DIR = "/root/cd-circuit-mechanism/results/exp03"
EXP00_RESULTS_DIR = "/root/cd-circuit-mechanism/results"

ALPHA = 1.0
NUM_EFFECTIVE = 25
NUM_INEFFECTIVE = 25
EXPERT_LAYERS = 42   # Gemma-2-9B
AMATEUR_LAYERS = 26  # Gemma-2-2B


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


def save_json(filename, data, results_dir=None):
    rdir = results_dir or RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)
    path = os.path.join(rdir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {path}")


def save_npy(filename, arr, results_dir=None):
    rdir = results_dir or RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)
    path = os.path.join(rdir, filename)
    np.save(path, arr)
    logger.info(f"Saved: {path}")


# ============================================================
# Sample selection from exp-00v2-fix results
# ============================================================

def load_exp00_results(exp00_dir, dataset_name, alpha):
    """Load per-sample results from exp-00v2-fix for given dataset and alpha."""
    filename = f"exp00v2fix_{dataset_name}_alpha{alpha}.json"
    path = os.path.join(exp00_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"exp-00v2-fix result not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data


def select_samples(exp00_dir, dataset_name, alpha, n_effective, n_ineffective):
    """Select CD-effective and CD-ineffective samples from exp-00v2-fix results.

    CD-effective: delta_logp > 0 (CD score > expert score)
    CD-ineffective: delta_logp <= 0

    For MC tasks, we use cd_logp_correct vs expert_logp_correct.
    Also loads alpha=0.0 results to get expert-only accuracy per sample.

    Returns dict with 'effective' and 'ineffective' lists, each containing
    sample dicts with keys: idx, prompt_info, correct, delta_logp, cd_score, expert_logp.
    """
    # Load CD results (alpha > 0)
    cd_data = load_exp00_results(exp00_dir, dataset_name, alpha)
    # Load expert-only results (alpha=0.0) for comparison
    expert_data = load_exp00_results(exp00_dir, dataset_name, 0.0)

    cd_samples = cd_data["per_sample"]
    expert_samples = expert_data["per_sample"]

    effective = []
    ineffective = []

    for i, (cd_s, exp_s) in enumerate(zip(cd_samples, expert_samples)):
        delta = cd_s.get("delta_logp", 0.0)
        # For MC: also check if CD flipped correctness
        cd_correct = cd_s.get("correct", False)
        exp_correct = exp_s.get("correct", False)

        entry = {
            "idx": i,
            "cd_correct": cd_correct,
            "expert_correct": exp_correct,
            "delta_logp": delta,
            "cd_score": cd_s.get("cd_logp_correct", cd_s.get("cd_score_correct", 0.0)),
            "expert_logp": cd_s.get("expert_logp_correct", 0.0),
        }

        if delta > 0:
            effective.append(entry)
        else:
            ineffective.append(entry)

    # Sort by magnitude of delta_logp (most extreme first)
    effective.sort(key=lambda x: -x["delta_logp"])
    ineffective.sort(key=lambda x: x["delta_logp"])

    n_eff = min(n_effective, len(effective))
    n_ineff = min(n_ineffective, len(ineffective))

    logger.info(f"[{dataset_name}] CD-effective: {len(effective)} total, selected {n_eff}")
    logger.info(f"[{dataset_name}] CD-ineffective: {len(ineffective)} total, selected {n_ineff}")

    return {
        "effective": effective[:n_eff],
        "ineffective": ineffective[:n_ineff],
        "dataset": dataset_name,
        "alpha": alpha,
    }


# ============================================================
# Prompt reconstruction from dataset
# ============================================================

def reconstruct_prompts(dataset_name, sample_indices, n_samples=200):
    """Reconstruct prompts and target tokens from the original datasets.

    Returns list of dicts with keys: idx, prompt, choices, correct_idx, correct_choice.
    """
    from datasets import load_dataset
    import random

    # NOTE: exp-00v2-fix uses a module-level RNG = random.Random(42) that advances
    # across all tasks (TruthfulQA, TriviaQA, GSM8K). Replicating its exact state is
    # fragile. Instead, we use per-sample deterministic seeds for shuffle and identify
    # the correct answer via label=1 (always valid regardless of shuffle order).

    if dataset_name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split=f"validation[:{n_samples}]")
        prompts = []
        for idx in sample_indices:
            item = ds[idx]
            choices = list(item['mc1_targets']['choices'])
            labels = list(item['mc1_targets']['labels'])
            correct_answer_text = choices[labels.index(1)]
            pairs = list(zip(choices, labels))
            # Per-sample deterministic seed: consistent across runs, unique per sample
            rng = random.Random(42 + idx)
            rng.shuffle(pairs)
            shuffled_choices = [p[0] for p in pairs]
            shuffled_labels = [p[1] for p in pairs]
            correct_idx = shuffled_labels.index(1)

            question = item['question']
            prompt_parts = [f"Q: {question}"]
            for j, c in enumerate(shuffled_choices):
                prompt_parts.append(f"({chr(65+j)}) {c}")
            prompt_parts.append("Answer: (")
            prompt = "\n".join(prompt_parts)

            prompts.append({
                "idx": idx,
                "prompt": prompt,
                "correct_choice": correct_answer_text,
                "correct_choice_letter": chr(65 + correct_idx),
                "correct_idx": correct_idx,
            })
        return prompts

    elif dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=f"validation[:{n_samples}]")
        prompts = []
        for idx in sample_indices:
            item = ds[idx]
            ctx = item['ctx']
            endings = item['endings']
            correct_idx = int(item['label'])

            prompt_parts = [f"Context: {ctx}"]
            for j, e in enumerate(endings):
                prompt_parts.append(f"({chr(65+j)}) {e}")
            prompt_parts.append("The most plausible continuation is (")
            prompt = "\n".join(prompt_parts)

            prompts.append({
                "idx": idx,
                "prompt": prompt,
                "correct_choice": endings[correct_idx],
                "correct_idx": correct_idx,
            })
        return prompts

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# ============================================================
# Per-layer logit extraction
# ============================================================

def get_layer_names(model):
    """Detect layer module names for hooking residual streams."""
    # Gemma-2 architecture: model.layers[i] with post-layernorm residual
    n_layers = model.config.num_hidden_layers
    return n_layers


def extract_layer_residuals(model, input_ids, device):
    """Run forward pass and extract residual stream at each layer.

    Uses output_hidden_states=True to get post-layer residuals.

    Returns: list of tensors, each shape (seq_len, hidden_dim), length = n_layers + 1
             (index 0 = embedding output, index l = after layer l-1)
    """
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
    # index 0 = embedding, index l = output of layer l-1
    hidden_states = outputs.hidden_states
    return hidden_states


def unembed_residual(model, hidden_state):
    """Project a hidden state through the unembedding (lm_head) to get logits.

    Args:
        model: HF CausalLM model
        hidden_state: tensor of shape (hidden_dim,) or (seq_len, hidden_dim)

    Returns: logits tensor of shape (vocab_size,) or (seq_len, vocab_size)
    """
    # Gemma-2 uses model.lm_head (nn.Linear, no bias)
    # Some models apply a final layernorm before unembedding
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        # Apply final RMSNorm before unembedding
        normed = model.model.norm(hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state.unsqueeze(0))
        normed = normed.squeeze(0)
    else:
        normed = hidden_state

    with torch.no_grad():
        logits = model.lm_head(normed)
    return logits


def compute_per_layer_logits(model, tokenizer, prompt, device, target_token_id=None):
    """Compute logits at each layer for the last token position.

    For each layer l (0 to n_layers):
      - Take residual stream at layer l, last token position
      - Project through unembedding to get logits
      - Extract logit for target_token_id (if provided) or return full logit vector

    Returns dict:
      - 'layer_logits': np.array of shape (n_layers+1,) if target_token_id given,
                        else (n_layers+1, vocab_size)
      - 'layer_deltas': np.array of shape (n_layers,) — per-layer contribution
                        (logit[l] - logit[l-1]) for target token
      - 'final_logits': np.array of shape (vocab_size,) — logits at final layer
      - 'n_layers': int
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    hidden_states = extract_layer_residuals(model, input_ids, device)
    n_layers = len(hidden_states) - 1  # subtract embedding layer

    last_pos = input_ids.shape[1] - 1
    layer_logits = []

    for l in range(n_layers + 1):
        h = hidden_states[l][0, last_pos, :]  # (hidden_dim,)
        logits_l = unembed_residual(model, h)  # (vocab_size,)

        if target_token_id is not None:
            layer_logits.append(logits_l[target_token_id].item())
        else:
            layer_logits.append(logits_l.cpu().float().numpy())

    # Clean up hidden states from GPU
    del hidden_states
    torch.cuda.empty_cache()

    layer_logits = np.array(layer_logits)

    # Compute per-layer deltas
    if target_token_id is not None:
        layer_deltas = np.diff(layer_logits)  # shape (n_layers,)
    else:
        layer_deltas = np.diff(layer_logits, axis=0)

    # Final logits (last hidden state through unembed, already computed)
    if target_token_id is not None:
        final_logit = layer_logits[-1]
    else:
        final_logit = layer_logits[-1]

    return {
        "layer_logits": layer_logits,
        "layer_deltas": layer_deltas,
        "final_logit": final_logit,
        "n_layers": n_layers,
    }


# ============================================================
# CD layer contribution analysis
# ============================================================

def analyze_sample(expert_model, amateur_model, tokenizer, prompt, target_token_id, alpha, device):
    """Full per-layer analysis for one sample.

    Returns dict with expert and amateur layer contributions and CD contribution.
    """
    # Expert per-layer analysis
    expert_result = compute_per_layer_logits(
        expert_model, tokenizer, prompt, device, target_token_id=target_token_id
    )

    # Amateur per-layer analysis
    amateur_result = compute_per_layer_logits(
        amateur_model, tokenizer, prompt, device, target_token_id=target_token_id
    )

    # CD at final layer
    expert_final = expert_result["final_logit"]
    amateur_final = amateur_result["final_logit"]
    cd_final_logit = expert_final - alpha * amateur_final

    return {
        "expert_layer_logits": expert_result["layer_logits"].tolist(),
        "expert_layer_deltas": expert_result["layer_deltas"].tolist(),
        "expert_n_layers": expert_result["n_layers"],
        "amateur_layer_logits": amateur_result["layer_logits"].tolist(),
        "amateur_layer_deltas": amateur_result["layer_deltas"].tolist(),
        "amateur_n_layers": amateur_result["n_layers"],
        "expert_final_logit": float(expert_final),
        "amateur_final_logit": float(amateur_final),
        "cd_final_logit": float(cd_final_logit),
        "alpha": alpha,
        "target_token_id": target_token_id,
    }


# ============================================================
# Aggregation and summary
# ============================================================

def aggregate_results(all_results, group_name):
    """Aggregate per-layer deltas across samples for one group (effective/ineffective).

    Returns dict with mean/std of layer deltas for expert and amateur.
    """
    expert_deltas = []
    amateur_deltas = []

    for r in all_results:
        expert_deltas.append(r["expert_layer_deltas"])
        amateur_deltas.append(r["amateur_layer_deltas"])

    if not expert_deltas:
        return {"group": group_name, "n_samples": 0}

    # Pad to max length (all expert should be same, all amateur should be same)
    expert_arr = np.array(expert_deltas)  # (n_samples, expert_layers)
    amateur_arr = np.array(amateur_deltas)  # (n_samples, amateur_layers)

    return {
        "group": group_name,
        "n_samples": len(all_results),
        "expert": {
            "n_layers": expert_arr.shape[1],
            "delta_mean": expert_arr.mean(axis=0).tolist(),
            "delta_std": expert_arr.std(axis=0).tolist(),
            "delta_abs_mean": np.abs(expert_arr).mean(axis=0).tolist(),
        },
        "amateur": {
            "n_layers": amateur_arr.shape[1],
            "delta_mean": amateur_arr.mean(axis=0).tolist(),
            "delta_std": amateur_arr.std(axis=0).tolist(),
            "delta_abs_mean": np.abs(amateur_arr).mean(axis=0).tolist(),
        },
    }


def compute_layer_importance(agg):
    """Rank layers by absolute mean delta (contribution magnitude)."""
    result = {}
    for model_name in ["expert", "amateur"]:
        if model_name not in agg or agg["n_samples"] == 0:
            continue
        abs_mean = agg[model_name]["delta_abs_mean"]
        ranked = sorted(enumerate(abs_mean), key=lambda x: -x[1])
        result[model_name] = {
            "ranking": [{"layer": l, "abs_mean_delta": float(v)} for l, v in ranked[:10]],
            "total_layers": len(abs_mean),
        }
    return result


# ============================================================
# Main pipeline
# ============================================================

def run_analysis(expert_model, amateur_model, tokenizer, selected_samples,
                 dataset_name, alpha, device):
    """Run per-layer analysis for all selected samples in one dataset."""

    all_indices = []
    sample_groups = {}
    for group in ["effective", "ineffective"]:
        for s in selected_samples[group]:
            all_indices.append(s["idx"])
            sample_groups[s["idx"]] = group

    if not all_indices:
        logger.warning(f"No samples for {dataset_name}")
        return None

    # Reconstruct prompts
    logger.info(f"Reconstructing {len(all_indices)} prompts for {dataset_name}...")
    prompts = reconstruct_prompts(dataset_name, all_indices)
    prompt_map = {p["idx"]: p for p in prompts}

    # Get target token IDs (first token of correct choice)
    target_tokens = {}
    for p in prompts:
        choice_text = p.get("correct_choice", p.get("correct_choice_letter", "A"))
        token_ids = tokenizer.encode(choice_text, add_special_tokens=False)
        target_tokens[p["idx"]] = token_ids[0] if token_ids else 0

    # Run analysis per sample
    effective_results = []
    ineffective_results = []
    per_sample_data = []

    for i, idx in enumerate(all_indices):
        group = sample_groups[idx]
        prompt = prompt_map[idx]["prompt"]
        target_token_id = target_tokens[idx]

        logger.info(f"  [{i+1}/{len(all_indices)}] idx={idx} group={group} "
                     f"target_token={tokenizer.decode([target_token_id])!r}")

        result = analyze_sample(
            expert_model, amateur_model, tokenizer,
            prompt, target_token_id, alpha, device
        )
        result["idx"] = idx
        result["group"] = group
        result["target_token_str"] = tokenizer.decode([target_token_id])

        per_sample_data.append(result)

        if group == "effective":
            effective_results.append(result)
        else:
            ineffective_results.append(result)

        if (i + 1) % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(f"  Progress: {i+1}/{len(all_indices)}, GPU mem: {mem:.1f} GB")

    # Aggregate
    eff_agg = aggregate_results(effective_results, "effective")
    ineff_agg = aggregate_results(ineffective_results, "ineffective")

    eff_importance = compute_layer_importance(eff_agg)
    ineff_importance = compute_layer_importance(ineff_agg)

    return {
        "dataset": dataset_name,
        "alpha": alpha,
        "n_effective": len(effective_results),
        "n_ineffective": len(ineffective_results),
        "per_sample": per_sample_data,
        "aggregate": {
            "effective": eff_agg,
            "ineffective": ineff_agg,
        },
        "layer_importance": {
            "effective": eff_importance,
            "ineffective": ineff_importance,
        },
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exp-03: Per-Layer Logit Diff Anatomy for Contrastive Decoding"
    )
    parser.add_argument("--expert-path", type=str, default=EXPERT_LOCAL,
                        help="Expert model path")
    parser.add_argument("--amateur-path", type=str, default=AMATEUR_LOCAL,
                        help="Amateur model path")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="CD alpha value (default: 1.0)")
    parser.add_argument("--num-effective", type=int, default=NUM_EFFECTIVE,
                        help="Number of CD-effective samples (default: 25)")
    parser.add_argument("--num-ineffective", type=int, default=NUM_INEFFECTIVE,
                        help="Number of CD-ineffective samples (default: 25)")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="Output directory")
    parser.add_argument("--exp00-results", type=str, default=EXP00_RESULTS_DIR,
                        help="exp-00v2-fix results directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device (default: cuda:0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with 2 samples per group")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["hellaswag", "truthfulqa"],
                        help="Datasets to analyze (default: hellaswag truthfulqa)")
    parser.add_argument("--serial-models", action="store_true",
                        help="Load models serially (lower VRAM, slower)")
    args = parser.parse_args()

    # Use args.results_dir throughout (passed to save functions)
    _results_dir = args.results_dir

    print_gpu_info()

    if not torch.cuda.is_available():
        logger.warning("No CUDA device. Running on CPU (will be very slow).")
        args.device = "cpu"

    n_eff = 2 if args.dry_run else args.num_effective
    n_ineff = 2 if args.dry_run else args.num_ineffective
    if args.dry_run:
        logger.info("*** DRY RUN MODE (2 samples per group) ***")

    # ==============================
    # Select samples from exp-00v2-fix
    # ==============================
    logger.info("="*60)
    logger.info("Phase 1: Selecting samples from exp-00v2-fix")
    logger.info("="*60)

    all_selected = {}
    for ds_name in args.datasets:
        try:
            selected = select_samples(
                args.exp00_results, ds_name, args.alpha, n_eff, n_ineff
            )
            all_selected[ds_name] = selected
        except FileNotFoundError as e:
            logger.error(f"Skipping {ds_name}: {e}")

    if not all_selected:
        logger.error("No datasets available. Ensure exp-00v2-fix results exist.")
        return

    # ==============================
    # Load models
    # ==============================
    logger.info("="*60)
    logger.info("Phase 2: Loading models")
    logger.info("="*60)

    logger.info(f"Loading expert model from {args.expert_path}...")
    expert_model = AutoModelForCausalLM.from_pretrained(
        args.expert_path, torch_dtype=torch.float16, device_map=args.device,
        output_hidden_states=True,
    )
    expert_model.eval()
    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"Expert loaded. GPU mem: {mem:.1f} GB")

    if args.serial_models:
        amateur_model = None
        logger.info("Serial mode: amateur will be loaded after expert processing.")
    else:
        logger.info(f"Loading amateur model from {args.amateur_path}...")
        amateur_model = AutoModelForCausalLM.from_pretrained(
            args.amateur_path, torch_dtype=torch.float16, device_map=args.device,
            output_hidden_states=True,
        )
        amateur_model.eval()
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        logger.info(f"Both models loaded. GPU mem: {mem:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(args.expert_path)

    # ==============================
    # Run per-layer analysis
    # ==============================
    logger.info("="*60)
    logger.info("Phase 3: Per-layer logit analysis")
    logger.info("="*60)

    for ds_name, selected in all_selected.items():
        logger.info(f"\n--- {ds_name.upper()} ---")

        if args.serial_models:
            # Serial mode: run expert first, then load amateur
            # For now, require both models loaded (serial mode placeholder)
            logger.warning("Serial mode not fully implemented. Loading amateur now.")
            amateur_model = AutoModelForCausalLM.from_pretrained(
                args.amateur_path, torch_dtype=torch.float16, device_map=args.device,
                output_hidden_states=True,
            )
            amateur_model.eval()

        result = run_analysis(
            expert_model, amateur_model, tokenizer,
            selected, ds_name, args.alpha, args.device
        )

        if result is None:
            continue

        # Save per-sample data
        save_json(f"exp03_{ds_name}_alpha{args.alpha}_persample.json", result, _results_dir)

        # Save numpy arrays for plotting
        if result["aggregate"]["effective"]["n_samples"] > 0:
            eff_expert_deltas = np.array([
                s["expert_layer_deltas"] for s in result["per_sample"]
                if s["group"] == "effective"
            ])
            eff_amateur_deltas = np.array([
                s["amateur_layer_deltas"] for s in result["per_sample"]
                if s["group"] == "effective"
            ])
            save_npy(f"exp03_{ds_name}_effective_expert_deltas.npy", eff_expert_deltas, _results_dir)
            save_npy(f"exp03_{ds_name}_effective_amateur_deltas.npy", eff_amateur_deltas, _results_dir)

        if result["aggregate"]["ineffective"]["n_samples"] > 0:
            ineff_expert_deltas = np.array([
                s["expert_layer_deltas"] for s in result["per_sample"]
                if s["group"] == "ineffective"
            ])
            ineff_amateur_deltas = np.array([
                s["amateur_layer_deltas"] for s in result["per_sample"]
                if s["group"] == "ineffective"
            ])
            save_npy(f"exp03_{ds_name}_ineffective_expert_deltas.npy", ineff_expert_deltas, _results_dir)
            save_npy(f"exp03_{ds_name}_ineffective_amateur_deltas.npy", ineff_amateur_deltas, _results_dir)

        # Print summary
        logger.info(f"\n--- {ds_name} SUMMARY ---")
        for group in ["effective", "ineffective"]:
            agg = result["aggregate"][group]
            if agg["n_samples"] == 0:
                continue
            importance = result["layer_importance"][group]
            logger.info(f"  {group}: {agg['n_samples']} samples")
            for model_name in ["expert", "amateur"]:
                if model_name in importance:
                    top3 = importance[model_name]["ranking"][:3]
                    top3_str = ", ".join(
                        f"L{r['layer']}({r['abs_mean_delta']:.3f})" for r in top3
                    )
                    logger.info(f"    {model_name} top-3 layers: {top3_str}")

    # Cleanup
    del expert_model
    if amateur_model is not None:
        del amateur_model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("\n" + "="*60)
    logger.info("EXP-03 COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
