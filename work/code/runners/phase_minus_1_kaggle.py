"""
Phase -1 完整运行脚本 — Kaggle Notebook 版
把所有代码合并到一个文件，直接粘进 Kaggle Cell 即可运行。

执行顺序：
  Cell 1: 本文件的全部代码（定义函数和类）
  Cell 2: main() 调用

预计运行时间：T4 GPU 上 5-10 分钟
"""

import math
import json
import time
import random
import re
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, replace


# ============================================================
# 第一部分：SeedBundle — 统一随机种子管理
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


DEFAULT_BUNDLE = SeedBundle()


# ============================================================
# 第二部分：信号计算函数
# ============================================================

def compute_entropy(logits: torch.Tensor) -> float:
    """
    logits: [vocab_size] float tensor
    returns: Shannon entropy in nats
    """
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum().item()
    return float(entropy)


def compute_margin(logits: torch.Tensor) -> float:
    """top1 和 top2 的 logit 差"""
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


def compute_max_prob(logits: torch.Tensor) -> float:
    """top1 的 softmax 概率（范围 0-1）"""
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    return float(F.softmax(logits, dim=-1).max().item())


def compute_neg_log_prob_correct(logits: torch.Tensor, correct_token_id: int) -> float:
    """
    正确答案 token 的负对数概率。用作"oracle"信号（需要知道答案）。
    不进 MMQ 主表，只用来诊断"信号理论上限"。
    """
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-log_probs[correct_token_id].item())


SIGNAL_REGISTRY = {
    "entropy": compute_entropy,
    "margin": compute_margin,
    "max_prob": compute_max_prob,
}


# ============================================================
# 第三部分：Bootstrap AUROC CI
# ============================================================

def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    scores: [N] 预测分数（高分=高置信）
    labels: [N] 0/1 正确性标签（1=正确）
    """
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
# 第四部分：单元测试
# ============================================================

def run_all_tests():
    """必须全部通过才能继续跑实验"""
    print("=" * 50)
    print("运行单元测试...")
    print("=" * 50)

    # test_entropy_uniform
    V = 100
    logits = torch.zeros(V)
    e = compute_entropy(logits)
    expected = math.log(V)
    assert abs(e - expected) < 1e-5, f"expected {expected}, got {e}"
    print(f"  ✅ test_entropy_uniform: H={e:.4f}, expected={expected:.4f}")

    # test_entropy_peaked
    logits = torch.zeros(100)
    logits[0] = 100.0
    e = compute_entropy(logits)
    assert e < 1e-3, f"expected near 0, got {e}"
    print(f"  ✅ test_entropy_peaked: H={e:.6f}")

    # test_entropy_two_peaks
    logits = torch.full((100,), -1000.0)
    logits[0] = 0.0
    logits[1] = 0.0
    e = compute_entropy(logits)
    expected = math.log(2)
    assert abs(e - expected) < 1e-4, f"expected {expected}, got {e}"
    print(f"  ✅ test_entropy_two_peaks: H={e:.4f}, expected={expected:.4f}")

    # test_entropy_1d_assert
    logits_2d = torch.zeros(1, 100)
    try:
        compute_entropy(logits_2d)
        assert False, "should have raised AssertionError"
    except AssertionError:
        print("  ✅ test_entropy_1d_assert: 2D logits correctly rejected")

    # test_margin
    logits = torch.tensor([10.0, 3.0, 1.0, -1.0, -5.0])
    m = compute_margin(logits)
    assert abs(m - 7.0) < 1e-5
    print(f"  ✅ test_margin: {m}")

    # test_max_prob
    logits = torch.full((100,), -100.0)
    logits[0] = 100.0
    p = compute_max_prob(logits)
    assert p > 0.99
    print(f"  ✅ test_max_prob: {p:.6f}")

    # test_max_prob_range
    logits = torch.randn(1000)
    p = compute_max_prob(logits)
    assert 0.0 <= p <= 1.0, f"max_prob out of range: {p}"
    print(f"  ✅ test_max_prob_range: {p:.4f} ∈ [0, 1]")

    # test_neg_log_prob_correct
    logits = torch.zeros(10)
    logits[3] = 5.0
    nlp = compute_neg_log_prob_correct(logits, 3)
    assert nlp > 0, f"neg_log_prob should be positive, got {nlp}"
    assert nlp < 5.0, f"neg_log_prob too large: {nlp}"
    print(f"  ✅ test_neg_log_prob_correct: {nlp:.4f}")

    # test_bootstrap_perfect_signal
    np.random.seed(42)
    scores = np.concatenate([np.random.randn(100) + 5, np.random.randn(100)])
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    r = bootstrap_auroc_ci(scores, labels, n_bootstrap=500)
    assert r["auroc_mean"] > 0.95, r
    assert r["ci_width"] < 0.1, r
    print(f"  ✅ test_bootstrap_perfect: AUROC={r['auroc_mean']:.3f}, "
          f"CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    # test_bootstrap_random_signal
    np.random.seed(0)
    scores = np.random.randn(500)
    labels = (np.random.rand(500) > 0.5).astype(int)
    r = bootstrap_auroc_ci(scores, labels, n_bootstrap=500)
    assert 0.4 < r["auroc_mean"] < 0.6, r
    print(f"  ✅ test_bootstrap_random: AUROC={r['auroc_mean']:.3f}, "
          f"CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    # test_bootstrap_degenerate
    np.random.seed(0)
    scores = np.random.randn(100)
    labels = np.ones(100)
    r = bootstrap_auroc_ci(scores, labels)
    assert r["degenerate"] == True
    print(f"  ✅ test_bootstrap_degenerate: {r['reason']}")

    print("\n🎉 所有 11 个测试通过！可以继续跑实验。\n")


# ============================================================
# 第五部分：Tokenizer 侦查
# ============================================================

def run_tokenizer_recon(model_id: str = "/kaggle/input/datasets/shizhenhso/metacognition-seed/0ai") -> dict:
    """
    H0-1: Tokenizer 侦查
    回答"SmolLM-135M 的答案 token 怎么切"
    """
    print("=" * 50)
    print("H0-1: Tokenizer 侦查")
    print("=" * 50)

    tok = AutoTokenizer.from_pretrained(model_id)

    candidates = {
        "single_digit": [str(i) for i in range(0, 10)],
        "two_digit": [str(i) for i in [10, 12, 19, 42, 77, 99]],
        "word_answers": ["yes", "no", "A", "B", "C", "true", "false",
                         "larger", "smaller", "equal"],
        "with_leading_space": [f" {i}" for i in range(0, 10)],
    }

    results = {"model": model_id, "categories": {}}

    for cat, items in candidates.items():
        cat_result = []
        for item in items:
            ids = tok.encode(item, add_special_tokens=False)
            decoded = [tok.decode([i]) for i in ids]
            cat_result.append({
                "text": item,
                "token_ids": ids,
                "token_count": len(ids),
                "decoded_tokens": decoded,
            })
        results["categories"][cat] = cat_result

    single_token_answers = []
    for cat, items in results["categories"].items():
        for r in items:
            if r["token_count"] == 1:
                single_token_answers.append({
                    "text": r["text"],
                    "category": cat,
                    "token_id": r["token_ids"][0]
                })
    results["single_token_answers"] = single_token_answers
    results["single_token_count"] = len(single_token_answers)

    test_prompts = [
        "Q: What is 2 + 3?\nA:",
        "Q: What is 2 + 3?\nA: ",
        "Question: 2+3=",
        "2+3=",
    ]
    prompt_analysis = []
    for p in test_prompts:
        ids = tok.encode(p, add_special_tokens=False)
        prompt_analysis.append({
            "prompt": p,
            "last_5_tokens": [tok.decode([i]) for i in ids[-5:]],
            "total_tokens": len(ids),
        })
    results["prompt_tail_analysis"] = prompt_analysis

    # 打印关键信息
    print(f"\n单 token 答案数: {results['single_token_count']}")
    print(f"\n前 20 个单 token 答案:")
    for s in results['single_token_answers'][:20]:
        print(f"  '{s['text']}' -> id={s['token_id']} (类别={s['category']})")
    print(f"\nprompt 尾部分析:")
    for p in results['prompt_tail_analysis']:
        print(f"  '{p['prompt']}' -> 尾5token: {p['last_5_tokens']}")

    # 判断标准
    single_digits = [r for r in results["categories"]["single_digit"] if r["token_count"] == 1]
    with_space = [r for r in results["categories"]["with_leading_space"] if r["token_count"] == 1]

    print(f"\n--- 判断 ---")
    print(f"  裸数字 0-9 单 token 数: {len(single_digits)}/10")
    print(f"  带空格数字 ' 0'-' 9' 单 token 数: {len(with_space)}/10")

    if len(single_digits) == 10:
        print("  ✅ 裸数字全部是单 token → 任务用个位数运算，无需前导空格")
    elif len(with_space) == 10:
        print("  ✅ 带空格数字全部是单 token → 必须用 'A: X' 格式（X前有空格）")
    else:
        print("  ⚠️ 部分数字不是单 token → 需要过滤非单 token 答案")

    return results


# ============================================================
# 第六部分：数据生成器
# ============================================================

def gen_add_small(n: int, seed: int = 42) -> list:
    """个位数加法，和 <= 9。答案保证是单 token 数字。"""
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
    """个位数减法，结果 >= 0 且 <= 9。"""
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
    """个位数比较大小：答案就是较大的那个数字（单 token）。"""
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
    """奇偶判断：答案 1=偶数 0=奇数（单 token 数字）。"""
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
# 第七部分：Phase -1 主运行器
# ============================================================

MODEL_ID = "/kaggle/input/datasets/shizhenhso/metacognition-seed/0ai"
N_SAMPLES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_with_leading_space(tokenizer, text: str) -> int:
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
        if i < 3:
            pred_token = tokenizer.decode([r["predicted_id"]])
            ans_token = tokenizer.decode([r["answer_token_id"]])
            print(f"  [{i}] prompt=...{s['prompt'][-20:]!r} "
                  f"answer={ans_token!r}(id={r['answer_token_id']}) "
                  f"predicted={pred_token!r}(id={r['predicted_id']}) "
                  f"correct={r['correct']}")
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
    """Phase -1 完整运行：侦查 → 测试 → 推理 → 决策"""

    # ---- Step 1: Tokenizer 侦查（H0-1，必须先跑） ----
    recon_results = run_tokenizer_recon(MODEL_ID)

    os.makedirs('/kaggle/working/results', exist_ok=True)
    with open('/kaggle/working/results/tokenizer_recon.json', 'w') as f:
        json.dump(recon_results, f, indent=2, ensure_ascii=False)
    print("  tokenizer_recon.json 已保存")

    single_count = recon_results["single_token_count"]
    if single_count < 10:
        print(f"\n🔴 单 token 答案只有 {single_count} 个，太少！不要继续。")
        return
    print(f"\n✅ 单 token 答案有 {single_count} 个，足够继续。\n")

    # ---- Step 2: 单元测试 ----
    run_all_tests()

    # ---- Step 3: 加载模型 ----
    print("=" * 50)
    print("加载模型")
    print("=" * 50)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Device: {DEVICE}")

    # ---- Step 4: 跑 Phase -1 ----
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

    # 保存结果
    out_path = '/kaggle/working/results/phase_minus_1_mini.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ---- Step 5: GATE -1 决策 ----
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

    # ---- Step 6: 冻结 gates.json hash ----
    print("\n" + "=" * 60)
    print("预注册检查")
    print("=" * 60)
    import hashlib
    gates_path = '/kaggle/working/gates.json'
    if os.path.exists(gates_path):
        with open(gates_path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        print(f"gates.json SHA256: {h}")
    else:
        print("⚠️ gates.json 不存在，需要上传")

    print(f"\n结果已保存到 {out_path}")
    print(f"Tokenizer 侦查已保存到 /kaggle/working/results/tokenizer_recon.json")
    print("\n请把 /kaggle/working/ 整个下载到本地做 git commit！")
