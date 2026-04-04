#!/usr/bin/env python3
"""
Exp-02: Circuit Tracing — Standard Decoding vs Contrastive Decoding
比较 Gemma-2-2B 在标准解码和 CD 下的 attribution graph 差异，
揭示 CD 在电路层面改变了什么。

核心思路：
- Gemma-2-2B 是 CD 中的 amateur model，也是 circuit-tracer 分析的主体
- 对比 CD 有效/无效 case 的 attribution graph 差异
- 找出被 CD 抑制的 transcoder features（即 amateur 的"有害电路"）
"""

import argparse
import json
import logging
import os
import gc
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# --- 配置 ---
MODEL_LOCAL = "/root/autodl-tmp/models/AI-ModelScope/gemma-2-2b"
MODEL_HF = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"  # GemmaScope PLTs for gemma-2-2b
RESULTS_DIR = "/root/cd-circuit-mechanism/results/exp02"
EXP00_RESULTS_DIR = "/root/cd-circuit-mechanism/results/exp00v2fix"

ALPHA = 1.0
NUM_SAMPLES = 50  # CD有效25 + CD无效25
TOP_K_FEATURES = 50
MAX_FEATURE_NODES = 8192
ATTRIBUTION_BATCH_SIZE = 256
MAX_N_LOGITS = 10

# --- 日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 加载 ReplacementModel
# ============================================================

def load_replacement_model(model_path, device="cuda", dtype=torch.bfloat16):
    """加载带 transcoder 的 ReplacementModel（基于 TransformerLens）。"""
    from circuit_tracer import ReplacementModel

    # 优先本地路径，fallback 到 HF
    if os.path.exists(model_path):
        model_name = model_path
    else:
        logger.warning(f"Local model not found at {model_path}, using HF: {MODEL_HF}")
        model_name = MODEL_HF

    logger.info(f"Loading ReplacementModel: {model_name} with transcoder_set='{TRANSCODER_SET}'")
    model = ReplacementModel.from_pretrained(
        model_name,
        TRANSCODER_SET,
        dtype=dtype,
        backend="transformerlens",
        device=device,
    )
    logger.info("ReplacementModel loaded.")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory used: {mem:.1f} GB")

    return model


# ============================================================
# 2. 从 exp-00v2-fix 结果中选取样本
# ============================================================

def select_samples(exp00_results_dir, alpha, num_cd_effective=25, num_cd_ineffective=25):
    """从 exp-00v2-fix 结果中选取 CD 有效/无效样本。

    CD 有效: cd_score > expert_score (即 delta_logp > 0 且 correct)
    CD 无效: cd_score <= expert_score (即 delta_logp <= 0 或 incorrect)

    只使用 MC 任务 (HellaSwag, TruthfulQA)，因为它们有明确的 per-sample scoring。
    """
    samples = {"cd_effective": [], "cd_ineffective": []}

    datasets_to_check = [
        (f"exp00v2fix_hellaswag_alpha{alpha}.json", "hellaswag"),
        (f"exp00v2fix_truthfulqa_alpha{alpha}.json", "truthfulqa"),
    ]

    # 同时加载 alpha=0.0 作为 expert-only baseline
    baseline_files = [
        (f"exp00v2fix_hellaswag_alpha0.0.json", "hellaswag"),
        (f"exp00v2fix_truthfulqa_alpha0.0.json", "truthfulqa"),
    ]

    baseline_data = {}
    for fname, ds_name in baseline_files:
        fpath = os.path.join(exp00_results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            baseline_data[ds_name] = {s["idx"]: s for s in data["per_sample"]}
            logger.info(f"Loaded baseline: {fname} ({len(data['per_sample'])} samples)")

    for fname, ds_name in datasets_to_check:
        fpath = os.path.join(exp00_results_dir, fname)
        if not os.path.exists(fpath):
            logger.warning(f"Results file not found: {fpath}")
            continue

        with open(fpath) as f:
            data = json.load(f)

        baseline_samples = baseline_data.get(ds_name, {})

        for s in data["per_sample"]:
            idx = s["idx"]
            baseline_s = baseline_samples.get(idx, {})

            # CD 有效: CD 下正确且 expert-only 下错误，或 delta_logp > 0
            cd_correct = s.get("correct", False)
            expert_correct = baseline_s.get("correct", False)
            delta_logp = s.get("delta_logp", 0.0)

            sample_info = {
                "dataset": ds_name,
                "idx": idx,
                "cd_correct": cd_correct,
                "expert_correct": expert_correct,
                "cd_score": s.get("cd_score_correct", 0.0),
                "expert_logp": s.get("expert_logp_correct", 0.0),
                "delta_logp": delta_logp,
            }

            # CD 有效: CD 改善了结果
            if cd_correct and (not expert_correct or delta_logp > 0):
                samples["cd_effective"].append(sample_info)
            # CD 无效: CD 没有改善或恶化了结果
            elif not cd_correct or delta_logp <= 0:
                samples["cd_ineffective"].append(sample_info)

    # 按 delta_logp 绝对值排序，选取最显著的样本
    samples["cd_effective"].sort(key=lambda x: x["delta_logp"], reverse=True)
    samples["cd_ineffective"].sort(key=lambda x: x["delta_logp"])

    selected_effective = samples["cd_effective"][:num_cd_effective]
    selected_ineffective = samples["cd_ineffective"][:num_cd_ineffective]

    logger.info(f"Selected samples: {len(selected_effective)} CD-effective, "
                f"{len(selected_ineffective)} CD-ineffective")

    return selected_effective, selected_ineffective


def load_prompt_for_sample(sample_info):
    """根据 dataset 和 idx 重建 prompt text。

    需要重新加载原始数据集来获取 prompt 文本。
    """
    ds_name = sample_info["dataset"]
    idx = sample_info["idx"]

    if ds_name == "hellaswag":
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation")
        item = ds[idx]
        prompt = item["ctx"]
        choices = item["endings"]
        correct_idx = int(item["label"])
        return {
            "prompt": prompt,
            "choices": choices,
            "correct_idx": correct_idx,
            "type": "mc",
        }

    elif ds_name == "truthfulqa":
        from datasets import load_dataset
        import random
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        item = ds[idx]
        choices = list(item['mc1_targets']['choices'])
        labels = list(item['mc1_targets']['labels'])
        correct_answer_text = choices[labels.index(1)]
        # Per-sample deterministic shuffle (consistent, unique per sample)
        pairs = list(zip(choices, labels))
        rng = random.Random(42 + idx)
        rng.shuffle(pairs)
        shuffled_choices = [p[0] for p in pairs]
        shuffled_labels = [p[1] for p in pairs]
        correct_idx = shuffled_labels.index(1)
        # NOTE: exp-00v2-fix scores each choice independently (per-choice logprob),
        # while exp-02 uses the raw question prompt for attribution. The correct
        # answer text is always identified via label=1 from the original dataset.
        prompt = f"Q: {item['question']}\nA: {correct_answer_text}"
        return {
            "prompt": prompt,
            "choices": shuffled_choices,
            "correct_idx": correct_idx,
            "correct_answer_text": correct_answer_text,
            "type": "mc",
        }

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


# ============================================================
# 3. 提取 Attribution Graph
# ============================================================

def extract_attribution_graph(model, prompt_text, save_path=None,
                              max_feature_nodes=MAX_FEATURE_NODES,
                              batch_size=ATTRIBUTION_BATCH_SIZE):
    """对单个 prompt 提取 attribution graph。

    Returns:
        graph: Graph 对象
        summary: dict 摘要（feature 统计等）
    """
    from circuit_tracer import attribute

    logger.info(f"Running attribution on prompt: '{prompt_text[:80]}...'")
    t0 = time.time()

    graph = attribute(
        prompt=prompt_text,
        model=model,
        max_n_logits=MAX_N_LOGITS,
        desired_logit_prob=0.95,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload="cpu",  # 节省 GPU 内存
        verbose=False,
    )

    elapsed = time.time() - t0
    logger.info(f"Attribution completed in {elapsed:.1f}s")

    # 提取摘要信息
    active_features = graph.active_features  # (n_active, 3): layer, pos, feature_idx
    n_active = active_features.shape[0] if active_features is not None else 0

    # 按层统计 active features
    layer_counts = defaultdict(int)
    if n_active > 0:
        for i in range(n_active):
            layer = int(active_features[i, 0])
            layer_counts[layer] += 1

    # 获取 logit targets
    logit_info = []
    if hasattr(graph, 'logit_targets') and graph.logit_targets is not None:
        for lt in graph.logit_targets:
            logit_info.append({
                "token": lt.token_str if hasattr(lt, 'token_str') else str(lt),
                "vocab_idx": int(lt.vocab_idx) if hasattr(lt, 'vocab_idx') else -1,
            })

    summary = {
        "n_active_features": n_active,
        "layer_feature_counts": dict(layer_counts),
        "logit_targets": logit_info,
        "attribution_time_s": elapsed,
    }

    # 保存 graph
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        graph.to_pt(save_path)
        logger.info(f"Graph saved to {save_path}")

    return graph, summary


# ============================================================
# 4. 提取 Feature Activations 和 Attribution 差异
# ============================================================

def extract_feature_activations(model, prompt_text):
    """提取 transcoder feature activations（不做 attribution，只做前向传播）。

    Returns:
        logits: (1, seq_len, vocab_size)
        activations: (n_layers, seq_len, d_transcoder) 稀疏或密集
    """
    logits, activations = model.get_activations(
        prompt_text,
        sparse=True,
        apply_activation_function=True,
    )
    return logits, activations


def compute_cd_attribution_diff(graph, model, prompt_text, alpha):
    """计算标准解码 vs CD 解码的 attribution 差异。

    思路：
    1. 从 graph 获取 active features 及其 activation values
    2. 获取 adjacency matrix 中各 feature 对 logit targets 的贡献
    3. CD logits = expert_logits - alpha * amateur_logits
       在 circuit 层面，amateur 的高激活 features 就是被 CD 抑制的部分

    Returns:
        diff_info: dict，包含每个 feature 的 attribution 和 CD 影响
    """
    active_features = graph.active_features  # (n_active, 3)
    adj_matrix = graph.adjacency_matrix  # (n_targets, n_sources)
    n_active = active_features.shape[0] if active_features is not None else 0

    if n_active == 0:
        return {"features": [], "n_active": 0}

    # 提取 feature activations
    activation_vals = graph.activation_values if hasattr(graph, 'activation_values') and graph.activation_values is not None else None

    # 计算每个 feature 对 logit 的贡献（adjacency matrix 的对应列）
    # adj_matrix 的列索引：前 n_active 列是 feature nodes
    feature_contributions = []
    n_logit_targets = len(graph.logit_targets) if hasattr(graph, 'logit_targets') and graph.logit_targets else 0

    for i in range(min(n_active, adj_matrix.shape[1])):
        layer = int(active_features[i, 0])
        pos = int(active_features[i, 1])
        feat_idx = int(active_features[i, 2])

        # 该 feature 对所有 target 的总贡献
        col = adj_matrix[:, i]
        total_effect = float(col.abs().sum())

        # 对 logit targets 的贡献（最后 n_logit 行）
        if n_logit_targets > 0:
            logit_effect = float(col[-n_logit_targets:].sum())
        else:
            logit_effect = total_effect

        act_val = float(activation_vals[i]) if activation_vals is not None and i < len(activation_vals) else 0.0

        feature_contributions.append({
            "layer": layer,
            "position": pos,
            "feature_idx": feat_idx,
            "activation": act_val,
            "total_effect": total_effect,
            "logit_effect": logit_effect,
            # CD 抑制量：activation * alpha（amateur 的 feature 被 CD 乘以 -alpha 抑制）
            "cd_suppression": act_val * alpha,
        })

    # 按 total_effect 排序
    feature_contributions.sort(key=lambda x: abs(x["total_effect"]), reverse=True)

    return {
        "features": feature_contributions,
        "n_active": n_active,
        "n_logit_targets": n_logit_targets,
    }


# ============================================================
# 5. 聚合分析
# ============================================================

def aggregate_analysis(all_results, top_k=TOP_K_FEATURES):
    """聚合所有样本的 attribution 差异，找出 CD 关键 features。

    分析维度：
    1. CD有效 vs CD无效 case 的 feature 激活差异
    2. 跨层的 attribution 分布差异
    3. Top-K differential features
    """
    # 收集所有 feature contributions
    effective_features = defaultdict(list)    # (layer, feat_idx) -> [effects]
    ineffective_features = defaultdict(list)

    effective_layer_effects = defaultdict(list)
    ineffective_layer_effects = defaultdict(list)

    for result in all_results:
        category = result["category"]  # "cd_effective" or "cd_ineffective"
        diff_info = result.get("attribution_diff", {})
        features = diff_info.get("features", [])

        for feat in features:
            key = (feat["layer"], feat["feature_idx"])
            effect = feat["total_effect"]
            activation = feat["activation"]

            if category == "cd_effective":
                effective_features[key].append({
                    "effect": effect,
                    "activation": activation,
                    "logit_effect": feat["logit_effect"],
                })
                effective_layer_effects[feat["layer"]].append(effect)
            else:
                ineffective_features[key].append({
                    "effect": effect,
                    "activation": activation,
                    "logit_effect": feat["logit_effect"],
                })
                ineffective_layer_effects[feat["layer"]].append(effect)

    # 找出 differential features：在 CD有效 case 中高激活但在 CD无效 case 中低激活的 features
    differential_features = []
    all_keys = set(list(effective_features.keys()) + list(ineffective_features.keys()))

    for key in all_keys:
        eff_effects = effective_features.get(key, [])
        ineff_effects = ineffective_features.get(key, [])

        eff_mean_act = np.mean([e["activation"] for e in eff_effects]) if eff_effects else 0.0
        ineff_mean_act = np.mean([e["activation"] for e in ineff_effects]) if ineff_effects else 0.0
        eff_mean_effect = np.mean([e["effect"] for e in eff_effects]) if eff_effects else 0.0
        ineff_mean_effect = np.mean([e["effect"] for e in ineff_effects]) if ineff_effects else 0.0

        differential_features.append({
            "layer": key[0],
            "feature_idx": key[1],
            "eff_mean_activation": float(eff_mean_act),
            "ineff_mean_activation": float(ineff_mean_act),
            "activation_diff": float(eff_mean_act - ineff_mean_act),
            "eff_mean_effect": float(eff_mean_effect),
            "ineff_mean_effect": float(ineff_mean_effect),
            "effect_diff": float(eff_mean_effect - ineff_mean_effect),
            "n_effective_samples": len(eff_effects),
            "n_ineffective_samples": len(ineff_effects),
        })

    # 按 activation_diff 绝对值排序
    differential_features.sort(key=lambda x: abs(x["activation_diff"]), reverse=True)
    top_differential = differential_features[:top_k]

    # 层级分析
    layer_analysis = {}
    all_layers = set(list(effective_layer_effects.keys()) + list(ineffective_layer_effects.keys()))
    for layer in sorted(all_layers):
        eff = effective_layer_effects.get(layer, [])
        ineff = ineffective_layer_effects.get(layer, [])
        layer_analysis[layer] = {
            "eff_mean_effect": float(np.mean(eff)) if eff else 0.0,
            "eff_total_effect": float(np.sum(eff)) if eff else 0.0,
            "ineff_mean_effect": float(np.mean(ineff)) if ineff else 0.0,
            "ineff_total_effect": float(np.sum(ineff)) if ineff else 0.0,
            "effect_diff": float(np.mean(eff) - np.mean(ineff)) if eff and ineff else 0.0,
            "n_eff_features": len(eff),
            "n_ineff_features": len(ineff),
        }

    return {
        "top_differential_features": top_differential,
        "layer_analysis": {str(k): v for k, v in layer_analysis.items()},
        "total_unique_features": len(all_keys),
        "n_effective_only": len(set(effective_features.keys()) - set(ineffective_features.keys())),
        "n_ineffective_only": len(set(ineffective_features.keys()) - set(effective_features.keys())),
        "n_shared": len(set(effective_features.keys()) & set(ineffective_features.keys())),
    }


# ============================================================
# 6. Main
# ============================================================

def setup_file_logging(results_dir):
    """添加文件日志 handler。"""
    os.makedirs(results_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(results_dir, "exp02.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)


def main():
    parser = argparse.ArgumentParser(
        description="Exp-02: Circuit Tracing — Standard vs Contrastive Decoding attribution analysis"
    )
    parser.add_argument("--model-path", type=str, default=MODEL_LOCAL,
                        help=f"Path to Gemma-2-2B model (default: {MODEL_LOCAL})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"CD alpha value (default: {ALPHA})")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help=f"Total number of samples (default: {NUM_SAMPLES})")
    parser.add_argument("--top-k", type=int, default=TOP_K_FEATURES,
                        help=f"Top-K differential features to report (default: {TOP_K_FEATURES})")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Output directory (default: {RESULTS_DIR})")
    parser.add_argument("--exp00-results", type=str, default=EXP00_RESULTS_DIR,
                        help=f"exp-00v2-fix results directory (default: {EXP00_RESULTS_DIR})")
    parser.add_argument("--max-feature-nodes", type=int, default=MAX_FEATURE_NODES,
                        help=f"Max feature nodes per attribution (default: {MAX_FEATURE_NODES})")
    parser.add_argument("--batch-size", type=int, default=ATTRIBUTION_BATCH_SIZE,
                        help=f"Attribution batch size (default: {ATTRIBUTION_BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run 2 samples to verify pipeline")
    parser.add_argument("--skip-attribution", action="store_true",
                        help="Skip attribution (only do sample selection, for debugging)")
    args = parser.parse_args()

    setup_file_logging(args.results_dir)

    logger.info("=" * 60)
    logger.info("Exp-02: Circuit Tracing — Standard vs CD Attribution")
    logger.info("=" * 60)
    logger.info(f"Config: alpha={args.alpha}, num_samples={args.num_samples}, "
                f"dry_run={args.dry_run}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Results dir: {args.results_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU: {gpu}, Memory: {mem:.1f} GB")

    # --- Step 1: 选取样本 ---
    logger.info("\n--- Step 1: Selecting samples from exp-00v2-fix ---")

    if args.dry_run:
        n_eff = 1
        n_ineff = 1
    else:
        n_eff = args.num_samples // 2
        n_ineff = args.num_samples - n_eff

    effective_samples, ineffective_samples = select_samples(
        args.exp00_results, args.alpha,
        num_cd_effective=n_eff,
        num_cd_ineffective=n_ineff,
    )

    if not effective_samples and not ineffective_samples:
        logger.error("No samples found! Check exp-00v2-fix results path.")
        logger.info("Generating synthetic test samples for pipeline validation...")

        # Fallback: 生成简单的测试 prompts
        effective_samples = [{
            "dataset": "synthetic",
            "idx": 0,
            "cd_correct": True,
            "expert_correct": False,
            "delta_logp": 0.5,
            "prompt_override": "The capital of France is",
        }]
        ineffective_samples = [{
            "dataset": "synthetic",
            "idx": 1,
            "cd_correct": False,
            "expert_correct": True,
            "delta_logp": -0.3,
            "prompt_override": "The largest planet in the solar system is",
        }]

    # 保存选取的样本
    sample_selection = {
        "alpha": args.alpha,
        "cd_effective": effective_samples,
        "cd_ineffective": ineffective_samples,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "sample_selection.json"), "w") as f:
        json.dump(sample_selection, f, indent=2)
    logger.info(f"Sample selection saved.")

    if args.skip_attribution:
        logger.info("--skip-attribution: stopping here.")
        return

    # --- Step 2: 加载模型 ---
    logger.info("\n--- Step 2: Loading ReplacementModel ---")
    model = load_replacement_model(args.model_path, device=device)

    # --- Step 3: 提取 attribution graphs ---
    logger.info("\n--- Step 3: Extracting attribution graphs ---")
    all_results = []

    all_samples = (
        [(s, "cd_effective") for s in effective_samples] +
        [(s, "cd_ineffective") for s in ineffective_samples]
    )

    for i, (sample, category) in enumerate(all_samples):
        logger.info(f"\n[{i+1}/{len(all_samples)}] {category} — "
                     f"dataset={sample['dataset']}, idx={sample['idx']}")

        # 获取 prompt
        if "prompt_override" in sample:
            prompt_data = {"prompt": sample["prompt_override"], "type": "synthetic"}
        else:
            try:
                prompt_data = load_prompt_for_sample(sample)
            except Exception as e:
                logger.error(f"Failed to load prompt: {e}")
                continue

        prompt_text = prompt_data["prompt"]

        # 提取 attribution graph
        graph_save_path = os.path.join(
            args.results_dir, "graphs",
            f"{category}_{sample['dataset']}_{sample['idx']}.pt"
        )

        try:
            graph, summary = extract_attribution_graph(
                model, prompt_text,
                save_path=graph_save_path,
                max_feature_nodes=args.max_feature_nodes,
                batch_size=args.batch_size,
            )

            # 计算 CD attribution 差异
            diff_info = compute_cd_attribution_diff(
                graph, model, prompt_text, args.alpha
            )

            result = {
                "sample": sample,
                "category": category,
                "prompt": prompt_text[:200],
                "graph_summary": summary,
                "attribution_diff": diff_info,
            }
            all_results.append(result)

            # 保存单个样本结果（不含大对象）
            sample_result_path = os.path.join(
                args.results_dir, "per_sample",
                f"{category}_{sample['dataset']}_{sample['idx']}.json"
            )
            os.makedirs(os.path.dirname(sample_result_path), exist_ok=True)
            with open(sample_result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Attribution failed for sample {sample['idx']}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --- Step 4: 聚合分析 ---
    logger.info(f"\n--- Step 4: Aggregate analysis ({len(all_results)} samples) ---")

    if all_results:
        agg = aggregate_analysis(all_results, top_k=args.top_k)

        # 输出 top differential features
        logger.info(f"\nTop-{args.top_k} Differential Features (CD effective vs ineffective):")
        logger.info(f"{'Layer':>6} {'FeatIdx':>8} {'EffAct':>10} {'IneffAct':>10} {'Diff':>10}")
        logger.info("-" * 50)
        for feat in agg["top_differential_features"][:20]:
            logger.info(f"{feat['layer']:>6} {feat['feature_idx']:>8} "
                        f"{feat['eff_mean_activation']:>10.4f} "
                        f"{feat['ineff_mean_activation']:>10.4f} "
                        f"{feat['activation_diff']:>10.4f}")

        # 层级分析
        logger.info(f"\nLayer-wise Attribution Analysis:")
        logger.info(f"{'Layer':>6} {'EffMean':>10} {'IneffMean':>10} {'Diff':>10}")
        logger.info("-" * 40)
        for layer_str in sorted(agg["layer_analysis"].keys(), key=int):
            la = agg["layer_analysis"][layer_str]
            logger.info(f"{layer_str:>6} {la['eff_mean_effect']:>10.4f} "
                        f"{la['ineff_mean_effect']:>10.4f} {la['effect_diff']:>10.4f}")

        # 保存聚合结果
        final_output = {
            "config": {
                "alpha": args.alpha,
                "num_samples": len(all_results),
                "model": args.model_path,
                "max_feature_nodes": args.max_feature_nodes,
                "dry_run": args.dry_run,
            },
            "aggregate": agg,
            "per_sample_summaries": [
                {
                    "category": r["category"],
                    "dataset": r["sample"]["dataset"],
                    "idx": r["sample"]["idx"],
                    "n_active_features": r["graph_summary"]["n_active_features"],
                    "top_features": r["attribution_diff"]["features"][:10],
                }
                for r in all_results
            ],
            "timestamp": datetime.now().isoformat(),
        }

        output_path = os.path.join(args.results_dir, "exp02_results.json")
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2, default=str)
        logger.info(f"\nFinal results saved to {output_path}")

    else:
        logger.warning("No results to aggregate!")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info("\n" + "=" * 60)
    logger.info("Exp-02 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
