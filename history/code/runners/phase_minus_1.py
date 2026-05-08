"""
Phase -1: 可行性底线
跑 SmolLM-135M on 4 tasks x 200 samples
输出每个任务的 (accuracy, entropy AUROC)

运行时间：T4 上约 5-10 分钟
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from common.seeding import SeedBundle
from signals.entropy import compute_entropy
from signals.margin import compute_margin
from signals.max_logit import compute_max_prob
from signals.signals import compute_neg_log_prob_correct
from core.stats import bootstrap_auroc_ci

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from generators import TASK_REGISTRY


MODEL_ID = "HuggingFaceTB/SmolLM-135M"
N_SAMPLES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_with_leading_space(tokenizer, text: str) -> int:
    """
    把答案 token 化。如果前面有空格版本是单 token，用带空格版本。
    返回 token_id 或 None（非单 token）。
    """
    ids_with_space = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids_with_space) == 1:
        return ids_with_space[0]
    ids_bare = tokenizer.encode(text, add_special_tokens=False)
    if len(ids_bare) == 1:
        return ids_bare[0]
    return None


@torch.no_grad()
def measure_one_sample(model, tokenizer, prompt: str, answer_token_id: int) -> dict:
    """
    对一个样本做一次 forward，返回所有信号 + 是否答对。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]

    predicted_id = last_logits.argmax().item()
    correct = int(predicted_id == answer_token_id)

    return {
        "entropy": compute_entropy(last_logits),
        "margin": compute_margin(last_logits),
        "max_prob": compute_max_prob(last_logits),
        "neg_log_prob_correct": compute_neg_log_prob_correct(last_logits, answer_token_id),
        "predicted_id": predicted_id,
        "correct": correct,
        "answer_token_id": answer_token_id,
    }


def run_task(task_name: str, model, tokenizer, n: int, seed: int) -> dict:
    print(f"\n=== Task: {task_name} ===")
    gen = TASK_REGISTRY[task_name]
    samples = gen(n, seed=seed)

    valid_samples = []
    for s in samples:
        tid = tokenize_with_leading_space(tokenizer, s["answer"])
        if tid is not None:
            s["answer_token_id"] = tid
            valid_samples.append(s)

    drop_rate = 1 - len(valid_samples) / len(samples)
    print(f"  原始 {len(samples)} 样本，保留 {len(valid_samples)} 个单 token 答案 "
          f"(丢弃率 {drop_rate:.1%})")

    if len(valid_samples) < 50:
        return {
            "task": task_name,
            "error": f"only {len(valid_samples)} valid samples, skipping",
        }

    t0 = time.time()
    records = []
    for i, s in enumerate(valid_samples):
        r = measure_one_sample(model, tokenizer, s["prompt"], s["answer_token_id"])
        r["meta"] = s["meta"]
        records.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(valid_samples)} done ({time.time()-t0:.1f}s)")

    correct = np.array([r["correct"] for r in records])
    accuracy = correct.mean()

    results = {
        "task": task_name,
        "n_total": len(samples),
        "n_valid": len(valid_samples),
        "drop_rate": drop_rate,
        "accuracy": float(accuracy),
        "time_seconds": time.time() - t0,
        "signals": {},
    }

    for sig_name in ["entropy", "margin", "max_prob"]:
        values = np.array([r[sig_name] for r in records])
        score = -values if sig_name == "entropy" else values
        ci = bootstrap_auroc_ci(score, correct, n_bootstrap=500)
        results["signals"][sig_name] = ci

    oracle_values = np.array([-r["neg_log_prob_correct"] for r in records])
    ci_oracle = bootstrap_auroc_ci(oracle_values, correct, n_bootstrap=500)
    results["signals"]["neg_log_prob_correct"] = ci_oracle

    print(f"  accuracy: {accuracy:.3f}")
    for sig, ci in results["signals"].items():
        if ci.get("degenerate"):
            print(f"  {sig}: DEGENERATE ({ci.get('reason')})")
        else:
            print(f"  {sig}: AUROC={ci['auroc_mean']:.3f} "
                  f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
                  f"width={ci['ci_width']:.3f}")

    return results


def main():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    bundle = SeedBundle()
    bundle.apply_global()

    all_results = {
        "model": MODEL_ID,
        "n_samples": N_SAMPLES,
        "device": DEVICE,
        "seed_bundle": {
            "data": bundle.data_seed,
            "sampling": bundle.sampling_seed,
        },
        "tasks": {},
    }

    for task_name in TASK_REGISTRY.keys():
        all_results["tasks"][task_name] = run_task(
            task_name, model, tokenizer, N_SAMPLES, bundle.data_seed
        )

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'phase_minus_1_mini.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("GATE -1 决策汇总")
    print("=" * 60)
    passed = []
    for task_name, r in all_results["tasks"].items():
        if "error" in r:
            status = f"❌ {r['error']}"
        else:
            acc = r["accuracy"]
            ent_ci = r["signals"].get("entropy", {})
            ent_auc = ent_ci.get("auroc_mean", float('nan'))
            if 0.40 <= acc <= 0.85:
                status = f"✅ acc={acc:.2f} ∈ [0.40, 0.85], entropy AUROC={ent_auc:.3f}"
                passed.append(task_name)
            elif acc < 0.40:
                status = f"⚠️ acc={acc:.2f} 太低（模型无法胜任）"
            else:
                status = f"⚠️ acc={acc:.2f} 太高（信号饱和）"
        print(f"  {task_name}: {status}")

    print(f"\n通过 GATE -1 的任务: {passed}")
    if len(passed) == 0:
        print("🔴 没有任务通过 → 考虑切换到 SmolLM-360M 或调难度")
    elif len(passed) == 1:
        print("🟡 只有 1 个任务通过 → 只用这个任务继续")
    else:
        print(f"🟢 {len(passed)} 个任务通过 → 可进入 Phase 0")

    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
