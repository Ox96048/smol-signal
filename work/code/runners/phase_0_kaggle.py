"""
Phase 0: MQ 仪器稳定性验证
1000 样本 × 4 任务 × 3 seed × 全部信号
目标：验证 Phase -1 的 AUROC 在大样本下是否稳定，CI 是否收窄到 < 0.05

GATE 0 通过条件：
  - ci_width_max < 0.05
  - sample_size_min >= 500

运行时间：T4 上约 30-50 分钟
"""

import math
import json
import time
import random
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, replace


# ============================================================
# SeedBundle
# ============================================================

@dataclass(frozen=True)
class SeedBundle:
    data_seed: int = 42
    sampling_seed: int = 43
    probe_init_seed: int = 44
    spl_init_seed: int = 45

    def apply_global(self):
        random.seed(self.data_seed)
        np.random.seed(self.data_seed)
        torch.manual_seed(self.data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.data_seed)

    def replace(self, **kwargs):
        return replace(self, **kwargs)


# ============================================================
# 信号函数
# ============================================================

def compute_entropy(logits: torch.Tensor) -> float:
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum().item()
    return float(entropy)


def compute_margin(logits: torch.Tensor) -> float:
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


def compute_max_prob(logits: torch.Tensor) -> float:
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    return float(F.softmax(logits, dim=-1).max().item())


def compute_neg_log_prob_correct(logits: torch.Tensor, correct_token_id: int) -> float:
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-log_probs[correct_token_id].item())


SIGNAL_REGISTRY = {
    "entropy": compute_entropy,
    "margin": compute_margin,
    "max_prob": compute_max_prob,
}


# ============================================================
# Bootstrap AUROC CI
# ============================================================

def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    assert len(labels) == n, "length mismatch"

    if len(np.unique(labels)) < 2:
        return {
            "auroc_mean": float('nan'),
            "ci_low": float('nan'),
            "ci_high": float('nan'),
            "ci_width": float('nan'),
            "n": n,
            "degenerate": True,
            "reason": "labels are all same class",
        }

    point_auroc = roc_auc_score(labels, scores)

    rng = np.random.RandomState(seed)
    boot_aurocs = []
    fails = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_aurocs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            fails += 1

    boot_aurocs = np.array(boot_aurocs)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_aurocs, alpha * 100)
    ci_high = np.percentile(boot_aurocs, (1 - alpha) * 100)

    return {
        "auroc_mean": float(point_auroc),
        "auroc_bootstrap_mean": float(boot_aurocs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_width": float(ci_high - ci_low),
        "n": n,
        "n_bootstrap_fails": fails,
        "degenerate": False,
        "direction_note": "auroc < 0.5 means signal predicts in reverse direction",
    }


# ============================================================
# 数据生成器（与 Phase -1 一致）
# ============================================================

def gen_add_small(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9 - a)
        ans = a + b
        out.append({
            "prompt": f"Q: What is {a} + {b}?\nA: ",
            "answer": str(ans),
            "meta": {"task": "add_small", "a": a, "b": b, "difficulty": "easy"}
        })
    return out


def gen_sub_small(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, a)
        ans = a - b
        out.append({
            "prompt": f"Q: What is {a} - {b}?\nA: ",
            "answer": str(ans),
            "meta": {"task": "sub_small", "a": a, "b": b, "difficulty": "easy"}
        })
    return out


def gen_compare(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        while a == b:
            b = rng.randint(1, 9)
        ans = max(a, b)
        out.append({
            "prompt": f"Q: Which is larger, {a} or {b}?\nA: ",
            "answer": str(ans),
            "meta": {"task": "compare", "a": a, "b": b, "difficulty": "medium"}
        })
    return out


def gen_yes_no_parity(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.randint(1, 99)
        ans = 1 if x % 2 == 0 else 0
        out.append({
            "prompt": f"Q: Is {x} even? 1=yes 0=no\nA: ",
            "answer": str(ans),
            "meta": {"task": "parity", "x": x, "difficulty": "medium"}
        })
    return out


TASK_REGISTRY = {
    "add_small": gen_add_small,
    "sub_small": gen_sub_small,
    "compare": gen_compare,
    "parity": gen_yes_no_parity,
}


# ============================================================
# Phase 0 主运行器
# ============================================================

MODEL_ID = "/kaggle/input/datasets/shizhenhso/metacognition-seed/0ai"
N_SAMPLES = 1000
N_SEEDS = 3
SEED_LIST = [42, 123, 456]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_answer(tokenizer, text: str) -> int:
    ids_bare = tokenizer.encode(text, add_special_tokens=False)
    if len(ids_bare) == 1:
        return ids_bare[0]
    ids_with_space = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids_with_space) == 1:
        return ids_with_space[0]
    return None


@torch.no_grad()
def measure_one_sample(model, tokenizer, prompt: str, answer_token_id: int) -> dict:
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


def run_task_one_seed(task_name: str, model, tokenizer, n: int, seed: int) -> list:
    gen = TASK_REGISTRY[task_name]
    samples = gen(n, seed=seed)

    valid_samples = []
    for s in samples:
        tid = tokenize_answer(tokenizer, s["answer"])
        if tid is not None:
            s["answer_token_id"] = tid
            valid_samples.append(s)

    records = []
    for i, s in enumerate(valid_samples):
        r = measure_one_sample(model, tokenizer, s["prompt"], s["answer_token_id"])
        r["meta"] = s["meta"]
        r["seed"] = seed
        records.append(r)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(valid_samples)} done")

    return records


def aggregate_across_seeds(all_records: list) -> dict:
    correct = np.array([r["correct"] for r in all_records])
    accuracy = correct.mean()

    results = {
        "n_total": len(all_records),
        "accuracy": float(accuracy),
        "n_correct": int(correct.sum()),
        "n_wrong": int(len(correct) - correct.sum()),
        "signals": {},
    }

    for sig_name in ["entropy", "margin", "max_prob"]:
        values = np.array([r[sig_name] for r in all_records])
        score = -values if sig_name == "entropy" else values
        ci = bootstrap_auroc_ci(score, correct, n_bootstrap=2000)
        results["signals"][sig_name] = ci

    oracle_values = np.array([-r["neg_log_prob_correct"] for r in all_records])
    ci_oracle = bootstrap_auroc_ci(oracle_values, correct, n_bootstrap=2000)
    results["signals"]["neg_log_prob_correct"] = ci_oracle

    return results


def main():
    print("=" * 60)
    print("Phase 0: MQ 仪器稳定性验证")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"N_SEEDS: {N_SEEDS}")
    print(f"SEED_LIST: {SEED_LIST}")
    print(f"Device: {DEVICE}")

    # 加载模型
    print(f"\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 运行所有任务 × 所有 seed
    all_results = {
        "phase": "phase_0",
        "model": MODEL_ID,
        "n_samples": N_SAMPLES,
        "n_seeds": N_SEEDS,
        "seed_list": SEED_LIST,
        "device": DEVICE,
        "tasks": {},
    }

    for task_name in TASK_REGISTRY.keys():
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")

        all_records = []
        for seed_idx, seed in enumerate(SEED_LIST):
            print(f"\n  Seed {seed_idx+1}/{N_SEEDS} (seed={seed}):")
            bundle = SeedBundle(data_seed=seed)
            bundle.apply_global()
            records = run_task_one_seed(task_name, model, tokenizer, N_SAMPLES, seed)
            acc = np.mean([r["correct"] for r in records])
            print(f"    accuracy={acc:.3f}, n={len(records)}")
            all_records.extend(records)

        # 跨 seed 汇总
        agg = aggregate_across_seeds(all_records)
        agg["seeds"] = {}
        for seed in SEED_LIST:
            seed_records = [r for r in all_records if r["seed"] == seed]
            seed_correct = np.array([r["correct"] for r in seed_records])
            agg["seeds"][str(seed)] = {
                "n": len(seed_records),
                "accuracy": float(seed_correct.mean()),
            }

        all_results["tasks"][task_name] = agg

        # 打印摘要
        print(f"\n  === {task_name} 汇总 (n={agg['n_total']}) ===")
        print(f"  accuracy: {agg['accuracy']:.3f}")
        for sig, ci in agg["signals"].items():
            if ci.get("degenerate"):
                print(f"  {sig}: DEGENERATE ({ci.get('reason')})")
            else:
                print(f"  {sig}: AUROC={ci['auroc_mean']:.3f} "
                      f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
                      f"width={ci['ci_width']:.3f}")

    # 保存结果
    os.makedirs('/kaggle/working/results', exist_ok=True)
    out_path = '/kaggle/working/results/phase_0.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # GATE 0 决策
    print(f"\n{'='*60}")
    print("GATE 0 决策汇总")
    print(f"{'='*60}")
    print(f"通过条件: ci_width < 0.05 且 n >= 500\n")

    gate0_passed = []
    for task_name, r in all_results["tasks"].items():
        acc = r["accuracy"]
        issues = []
        for sig_name in ["entropy", "margin", "max_prob"]:
            ci = r["signals"].get(sig_name, {})
            if ci.get("degenerate"):
                issues.append(f"{sig_name}: DEGENERATE")
                continue
            width = ci["ci_width"]
            if width > 0.05:
                issues.append(f"{sig_name}: ci_width={width:.3f} > 0.05")

        n = r["n_total"]
        if n < 500:
            issues.append(f"n={n} < 500")

        if not issues:
            status = f"✅ PASS (acc={acc:.3f})"
            gate0_passed.append(task_name)
        else:
            status = f"⚠️ {'; '.join(issues)}"

        print(f"  {task_name}: {status}")

    print(f"\n通过 GATE 0 的任务: {gate0_passed}")
    if len(gate0_passed) >= 2:
        print(f"🟢 {len(gate0_passed)} 个任务通过 → 可进入 Phase 1")
    elif len(gate0_passed) == 1:
        print("🟡 只有 1 个任务通过 → 谨慎进入 Phase 1")
    else:
        print("🔴 没有任务通过 → 需要增加样本量或换模型")

    # gates.json hash 校验
    print(f"\n{'='*60}")
    print("预注册校验")
    print(f"{'='*60}")
    gates_path = '/kaggle/working/gates.json'
    if os.path.exists(gates_path):
        with open(gates_path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        print(f"gates.json SHA256: {h}")
        print(f"预期 SHA256: efda24d5634b33b189a0b52ed309b3021a8a5943293773036343423c73ab3d4f")
        if h == "efda24d5634b33b189a0b52ed309b3021a8a5943293773036343423c73ab3d4f":
            print("✅ gates.json 未被篡改")
        else:
            print("⚠️ gates.json SHA256 不匹配！文件可能被修改过")
    else:
        print("⚠️ gates.json 不存在，需要上传")

    print(f"\n结果已保存到 {out_path}")
    print(f"总运行时间: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
