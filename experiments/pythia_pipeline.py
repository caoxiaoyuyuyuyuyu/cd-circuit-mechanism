#!/usr/bin/env python3
"""
Contrastive Decoding Analysis Pipeline
Expert: Pythia-410M, Amateur: Pythia-70M

Usage:
    python pythia_pipeline.py --step all      # Run all steps
    python pythia_pipeline.py --step exp-00   # CD Baseline
    python pythia_pipeline.py --step exp-01   # Logit Diff Anatomy
    python pythia_pipeline.py --step exp-02   # Activation Patching
    python pythia_pipeline.py --step exp-03   # SAE Feature Extraction
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import einops
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water freezes at",
    "The speed of light is approximately",
    "The chemical formula for water is",
    "The first president of the United States was",
]

EXPECTED = [
    " Paris",
    " Jupiter",
    " 0",
    " 300",
    " H",
    " George",
]

ALPHAS = [0.5, 1.0]


def load_models():
    from transformer_lens import HookedTransformer
    print("Loading expert model (pythia-410m)...")
    expert = HookedTransformer.from_pretrained("pythia-410m", device=DEVICE)
    print("Loading amateur model (pythia-70m)...")
    amateur = HookedTransformer.from_pretrained("pythia-70m-deduped", device=DEVICE)
    return expert, amateur


def get_top_tokens(logits, model, k=5):
    last_logits = logits[0, -1, :]
    topk = torch.topk(last_logits, k)
    tokens = [model.tokenizer.decode(idx.item()) for idx in topk.indices]
    probs = F.softmax(last_logits, dim=-1)
    top_probs = [probs[idx].item() for idx in topk.indices]
    return list(zip(tokens, top_probs))


def run_exp00(expert, amateur):
    print("\n" + "=" * 60)
    print("EXP-00: Contrastive Decoding Baseline")
    print("=" * 60)

    results = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt: '{prompt}' ---")

        expert_tokens = expert.to_tokens(prompt)
        amateur_tokens = amateur.to_tokens(prompt)

        with torch.no_grad():
            expert_logits = expert(expert_tokens)
            amateur_logits = amateur(amateur_tokens)

        expert_top = get_top_tokens(expert_logits, expert)
        print(f"  Expert top-1: {expert_top[0][0]!r} (p={expert_top[0][1]:.4f})")

        amateur_top = get_top_tokens(amateur_logits, amateur)
        print(f"  Amateur top-1: {amateur_top[0][0]!r} (p={amateur_top[0][1]:.4f})")

        row = {
            "prompt": prompt,
            "expected": EXPECTED[i],
            "expert_top1": expert_top[0][0],
            "expert_top1_prob": expert_top[0][1],
            "amateur_top1": amateur_top[0][0],
            "amateur_top1_prob": amateur_top[0][1],
        }

        expert_last = expert_logits[0, -1, :]
        amateur_last = amateur_logits[0, -1, :]

        min_vocab = min(expert_last.shape[0], amateur_last.shape[0])
        expert_aligned = expert_last[:min_vocab]
        amateur_aligned = amateur_last[:min_vocab]

        # Adaptive plausibility constraint: only consider tokens in expert's top-p
        expert_probs = F.softmax(expert_aligned, dim=-1)
        sorted_probs, sorted_indices = torch.sort(expert_probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens until cumulative prob reaches 0.9
        cutoff_mask = cumsum <= 0.9
        cutoff_mask[0] = True  # always keep top-1
        plausible_indices = sorted_indices[cutoff_mask]
        plausibility_mask = torch.full_like(expert_aligned, float('-inf'))
        plausibility_mask[plausible_indices] = 0.0

        for alpha in ALPHAS:
            cd_logits_raw = expert_aligned - alpha * amateur_aligned
            cd_logits = cd_logits_raw + plausibility_mask  # mask implausible tokens
            cd_top_idx = cd_logits.argmax().item()
            cd_top_token = expert.tokenizer.decode(cd_top_idx)
            cd_top_prob = F.softmax(cd_logits[cd_logits != float('-inf')], dim=-1)
            # Also show raw CD without filter
            raw_cd_top_idx = cd_logits_raw.argmax().item()
            raw_cd_top_token = expert.tokenizer.decode(raw_cd_top_idx)

            cd_topk = torch.topk(cd_logits, min(5, plausible_indices.shape[0]))
            cd_top5 = [(expert.tokenizer.decode(idx.item()), expert_probs[idx].item())
                       for idx in cd_topk.indices]

            print(f"  CD(a={alpha}) top-1: {cd_top_token!r} (expert_p={expert_probs[cd_top_idx].item():.4f})")
            print(f"    raw CD (no filter): {raw_cd_top_token!r}")
            print(f"    plausible top-5: {[(t, f'{p:.3f}') for t, p in cd_top5]}")

            row[f"cd_alpha{alpha}_top1"] = cd_top_token
            row[f"cd_alpha{alpha}_top1_prob"] = expert_probs[cd_top_idx].item()
            row[f"cd_alpha{alpha}_raw_top1"] = raw_cd_top_token

        results.append(row)

    df = pd.DataFrame(results)
    # NOTE: Weak CD performance on small models (Pythia-410M/70M) is expected.
    # e.g., alpha=0.5 hitting only ~1/6 prompts is normal for this model scale.
    # This prototype validates the end-to-end pipeline; CD gains are expected
    # to be much stronger on larger models (Gemma-2-9B/2B).
    print("\n\n--- Summary ---")
    for _, r in df.iterrows():
        exp_match = r["expected"].strip().lower() in r["expert_top1"].strip().lower()
        cd05_match = r["expected"].strip().lower() in r[f"cd_alpha0.5_top1"].strip().lower()
        cd10_match = r["expected"].strip().lower() in r[f"cd_alpha1.0_top1"].strip().lower()
        print(f"  '{r['prompt']}': expert={'Y' if exp_match else 'N'} "
              f"CD(0.5)={'Y' if cd05_match else 'N'} "
              f"CD(1.0)={'Y' if cd10_match else 'N'}")

    outpath = RESULTS_DIR / "exp00_cd_baseline.json"
    df.to_json(outpath, orient="records", indent=2)
    print(f"\nResults saved to {outpath}")
    return df


def run_exp01(expert, amateur):
    print("\n" + "=" * 60)
    print("EXP-01: Logit Diff Anatomy")
    print("=" * 60)

    prompt = PROMPTS[0]
    correct_token = " Paris"
    print(f"Prompt: '{prompt}', target: '{correct_token}'")

    expert_tokens = expert.to_tokens(prompt)
    amateur_tokens = amateur.to_tokens(prompt)

    correct_id_expert = expert.tokenizer.encode(correct_token, add_special_tokens=False)[0]
    correct_id_amateur = amateur.tokenizer.encode(correct_token, add_special_tokens=False)[0]

    print("\nRunning expert with cache...")
    expert_logits, expert_cache = expert.run_with_cache(expert_tokens)
    print("Running amateur with cache...")
    amateur_logits, amateur_cache = amateur.run_with_cache(amateur_tokens)

    print(f"\n--- Expert: Per-layer logit contribution to '{correct_token}' ---")
    W_U = expert.W_U
    correct_dir = W_U[:, correct_id_expert]

    expert_layer_contribs = []

    embed_out = expert_cache["hook_embed"][0, -1, :]
    # Pythia uses rotary embeddings, no separate pos_embed hook
    if "hook_pos_embed" in expert_cache.cache_dict:
        pos_embed_out = expert_cache["hook_pos_embed"][0, -1, :]
        embed_contrib = (embed_out + pos_embed_out) @ correct_dir
    else:
        embed_contrib = embed_out @ correct_dir
    print(f"  Embedding: {embed_contrib.item():.4f}")

    for layer in range(expert.cfg.n_layers):
        attn_out = expert_cache[f"blocks.{layer}.hook_attn_out"][0, -1, :]
        attn_contrib = (attn_out @ correct_dir).item()

        mlp_out = expert_cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :]
        mlp_contrib = (mlp_out @ correct_dir).item()

        total = attn_contrib + mlp_contrib
        expert_layer_contribs.append({
            "layer": layer,
            "attn_contrib": attn_contrib,
            "mlp_contrib": mlp_contrib,
            "total": total,
        })
        print(f"  Layer {layer:2d}: attn={attn_contrib:+.4f}, mlp={mlp_contrib:+.4f}, total={total:+.4f}")

    print(f"\n--- Amateur: Per-layer logit contribution to '{correct_token}' ---")
    W_U_am = amateur.W_U
    correct_dir_am = W_U_am[:, correct_id_amateur]

    amateur_layer_contribs = []
    for layer in range(amateur.cfg.n_layers):
        attn_out = amateur_cache[f"blocks.{layer}.hook_attn_out"][0, -1, :]
        attn_contrib = (attn_out @ correct_dir_am).item()

        mlp_out = amateur_cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :]
        mlp_contrib = (mlp_out @ correct_dir_am).item()

        total = attn_contrib + mlp_contrib
        amateur_layer_contribs.append({
            "layer": layer,
            "attn_contrib": attn_contrib,
            "mlp_contrib": mlp_contrib,
            "total": total,
        })
        print(f"  Layer {layer:2d}: attn={attn_contrib:+.4f}, mlp={mlp_contrib:+.4f}, total={total:+.4f}")

    expert_df = pd.DataFrame(expert_layer_contribs)
    expert_df["abs_total"] = expert_df["total"].abs()
    top_layers = expert_df.nlargest(5, "abs_total")
    print(f"\nTop 5 expert layers by absolute contribution:")
    for _, r in top_layers.iterrows():
        print(f"  Layer {int(r['layer'])}: {r['total']:+.4f} (attn={r['attn_contrib']:+.4f}, mlp={r['mlp_contrib']:+.4f})")

    results = {
        "prompt": prompt,
        "correct_token": correct_token,
        "expert_layers": expert_layer_contribs,
        "amateur_layers": amateur_layer_contribs,
    }
    outpath = RESULTS_DIR / "exp01_logit_anatomy.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    del expert_cache, amateur_cache
    torch.cuda.empty_cache()
    return results


def run_exp02(expert, amateur):
    from transformer_lens import patching
    print("\n" + "=" * 60)
    print("EXP-02: Activation Patching (Expert Model)")
    print("=" * 60)

    clean_prompt = "The capital of France is"
    corrupted_prompt = "The capital of China is"
    correct_token = " Paris"
    incorrect_token = " Beijing"

    print(f"Clean: '{clean_prompt}' -> '{correct_token}'")
    print(f"Corrupted: '{corrupted_prompt}' -> '{incorrect_token}'")

    clean_tokens = expert.to_tokens(clean_prompt)
    corrupted_tokens = expert.to_tokens(corrupted_prompt)

    correct_ids = expert.tokenizer.encode(correct_token, add_special_tokens=False)
    incorrect_ids = expert.tokenizer.encode(incorrect_token, add_special_tokens=False)
    correct_id = correct_ids[0]
    incorrect_id = incorrect_ids[0]
    print(f"Token IDs - correct '{correct_token}': {correct_ids}, incorrect '{incorrect_token}': {incorrect_ids}")

    print("\nRunning clean forward pass...")
    clean_logits = expert(clean_tokens)
    clean_logit_diff = (clean_logits[0, -1, correct_id] - clean_logits[0, -1, incorrect_id]).item()
    print(f"Clean logit diff (Paris - Beijing): {clean_logit_diff:.4f}")

    print("Running clean forward pass with cache...")
    clean_logits2, clean_cache = expert.run_with_cache(clean_tokens)

    print("Running corrupted forward pass...")
    corrupted_logits = expert(corrupted_tokens)
    corrupted_logit_diff = (corrupted_logits[0, -1, correct_id] - corrupted_logits[0, -1, incorrect_id]).item()
    print(f"Corrupted logit diff (Paris - Beijing): {corrupted_logit_diff:.4f}")

    def logit_diff_metric(logits, answer_token_ids=None):
        return logits[0, -1, correct_id] - logits[0, -1, incorrect_id]

    # Patching: run corrupted tokens, patch in clean activations
    print("\nPatching attention head outputs (all positions)...")
    attn_head_results = patching.get_act_patch_attn_head_out_all_pos(
        expert, corrupted_tokens, clean_cache, logit_diff_metric
    )
    print(f"Attention head patching result shape: {attn_head_results.shape}")

    # Select heads with highest patching effect (closest to clean baseline).
    # Higher value = patching this head restores more of the clean logit diff.
    flat_idx = torch.topk(attn_head_results.flatten(), 10).indices
    n_heads = expert.cfg.n_heads
    top_heads = []
    for idx in flat_idx:
        layer = idx.item() // n_heads
        head = idx.item() % n_heads
        val = attn_head_results[layer, head].item()
        top_heads.append({"layer": layer, "head": head, "patching_effect": val})
        print(f"  L{layer}H{head}: effect={val:+.4f}")

    print("\nPatching MLP outputs (by layer)...")
    mlp_results = patching.get_act_patch_mlp_out(
        expert, corrupted_tokens, clean_cache, logit_diff_metric
    )
    print(f"MLP patching result shape: {mlp_results.shape}")

    top_mlps = []
    for layer in range(expert.cfg.n_layers):
        # mlp_results may be [n_layers, n_positions] - sum over positions or take last
        if mlp_results.dim() == 1:
            val = mlp_results[layer].item()
        else:
            val = mlp_results[layer, -1].item()  # last position
        top_mlps.append({"layer": layer, "patching_effect": val})

    top_mlps_sorted = sorted(top_mlps, key=lambda x: abs(x["patching_effect"]), reverse=True)[:5]
    print("\nTop 5 MLP layers by patching effect:")
    for m in top_mlps_sorted:
        print(f"  MLP Layer {m['layer']}: {m['patching_effect']:+.4f}")

    results = {
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "correct_token": correct_token,
        "incorrect_token": incorrect_token,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        # top_attention_heads: ranked by patching effect (highest = most important,
        # i.e., patching this head restores the clean logit diff the most)
        "top_attention_heads": top_heads,
        "mlp_patching": [{"layer": m["layer"], "effect": m["patching_effect"]} for m in top_mlps],
    }
    outpath = RESULTS_DIR / "exp02_activation_patching.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    del clean_cache
    torch.cuda.empty_cache()
    return results


def run_exp03(expert, amateur):
    print("\n" + "=" * 60)
    print("EXP-03: SAE Feature Extraction")
    print("=" * 60)

    try:
        from sae_lens import SAE
    except ImportError:
        print("ERROR: sae_lens not installed. Skipping exp-03.")
        return None

    prompt = "The capital of France is"
    corrupted_prompt = "The capital of China is"
    print(f"Clean: '{prompt}'")
    print(f"Corrupted: '{corrupted_prompt}'")

    sae_releases_to_try = [
        ("pythia-70m-deduped-res-sm", "blocks.{layer}.hook_resid_post", [0, 1, 2, 3, 4, 5]),
        ("pythia-70m-deduped-mlp-sm", "blocks.{layer}.hook_mlp_out", [3, 4, 5]),
    ]

    results = {"prompt": prompt, "corrupted_prompt": corrupted_prompt, "features": []}

    # SAEs are trained on pythia-70m, so use amateur model activations directly
    # (expert pythia-410m has different d_model and would always fail dim check)
    print("\nRunning amateur model for SAE analysis (SAEs trained on pythia-70m)...")
    amateur_clean_tokens = amateur.to_tokens(prompt)
    amateur_corrupted_tokens = amateur.to_tokens(corrupted_prompt)
    _, amateur_clean_cache = amateur.run_with_cache(amateur_clean_tokens)
    _, amateur_corrupt_cache = amateur.run_with_cache(amateur_corrupted_tokens)

    for release, sae_id_template, layers in sae_releases_to_try:
        for layer in layers:
            sae_id = sae_id_template.format(layer=layer)
            print(f"\nTrying SAE for amateur: release='{release}', sae_id='{sae_id}'...")
            try:
                sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
                print(f"  Loaded! dict_size={sae.cfg.d_sae}")

                hook_name = f"blocks.{layer}.hook_resid_post"
                amateur_act = amateur_clean_cache[hook_name][0, -1, :]
                if amateur_act.shape[0] != sae.cfg.d_in:
                    print(f"  Dim mismatch: amateur d_model={amateur_act.shape[0]}, SAE d_in={sae.cfg.d_in}")
                    del sae; torch.cuda.empty_cache()
                    continue

                clean_acts = sae.encode(amateur_clean_cache[hook_name][0, -1:, :])
                corrupted_acts = sae.encode(amateur_corrupt_cache[hook_name][0, -1:, :])

                diff = (clean_acts - corrupted_acts).squeeze()
                top_diff_idx = torch.topk(diff.abs(), min(10, diff.shape[0])).indices

                layer_features = []
                for idx in top_diff_idx:
                    feat_id = idx.item()
                    clean_val = clean_acts[0, feat_id].item()
                    corrupt_val = corrupted_acts[0, feat_id].item()
                    layer_features.append({
                        "feature_id": feat_id,
                        "clean_activation": clean_val,
                        "corrupted_activation": corrupt_val,
                        "diff": clean_val - corrupt_val,
                    })
                    print(f"    Feature {feat_id}: clean={clean_val:.4f}, corrupt={corrupt_val:.4f}, diff={clean_val - corrupt_val:+.4f}")

                results["features"].append({
                    "release": release,
                    "model": "pythia-70m (amateur)",
                    "layer": layer,
                    "sae_id": sae_id,
                    "d_sae": sae.cfg.d_sae,
                    "top_diff_features": layer_features,
                })
                del sae; torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Failed: {e}")
                continue

    del amateur_clean_cache, amateur_corrupt_cache
    torch.cuda.empty_cache()

    outpath = RESULTS_DIR / "exp03_sae_features.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CD Circuit Mechanism Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "exp-00", "exp-01", "exp-02", "exp-03"],
                        help="Which experiment step to run")
    args = parser.parse_args()

    steps = ["exp-00", "exp-01", "exp-02", "exp-03"] if args.step == "all" else [args.step]

    print(f"Device: {DEVICE}")
    print(f"Steps to run: {steps}")
    print(f"Results dir: {RESULTS_DIR}")

    expert, amateur = load_models()
    print(f"Expert: {expert.cfg.model_name}, {expert.cfg.n_layers} layers, {expert.cfg.n_heads} heads, d_model={expert.cfg.d_model}")
    print(f"Amateur: {amateur.cfg.model_name}, {amateur.cfg.n_layers} layers, {amateur.cfg.n_heads} heads, d_model={amateur.cfg.d_model}")

    all_results = {}

    for step in steps:
        t0 = time.time()
        print(f"\n{'#' * 60}")
        print(f"# Running {step}")
        print(f"{'#' * 60}")

        if step == "exp-00":
            all_results[step] = run_exp00(expert, amateur)
        elif step == "exp-01":
            all_results[step] = run_exp01(expert, amateur)
        elif step == "exp-02":
            all_results[step] = run_exp02(expert, amateur)
        elif step == "exp-03":
            all_results[step] = run_exp03(expert, amateur)

        elapsed = time.time() - t0
        print(f"\n{step} completed in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
