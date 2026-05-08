"""
诊断脚本：验证小模型的 prompt/answer/predicted 对齐情况

对每个任务打印：
  1. prompt 原文
  2. 期望答案 token (id, text)
  3. 模型预测 token (id, text)
  4. Top-5 预测及其概率
  5. 所有信号值
  6. 是否正确

每个任务打印 10 个样本（5 正确 + 5 错误，如果有的话）

运行时间：T4 上约 1-2 分钟
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "/kaggle/input/datasets/shizhenhso/metacognition-seed/0ai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DIAG = 20


def gen_add_small(n, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9 - a)
        ans = a + b
        out.append({
            "prompt": f"Q: What is {a} + {b}?\nA: ",
            "answer": str(ans),
            "meta": {"task": "add_small", "a": a, "b": b}
        })
    return out


def gen_sub_small(n, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, a)
        ans = a - b
        out.append({
            "prompt": f"Q: What is {a} - {b}?\nA: ",
            "answer": str(ans),
            "meta": {"task": "sub_small", "a": a, "b": b}
        })
    return out


def gen_compare(n, seed=42):
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
            "meta": {"task": "compare", "a": a, "b": b}
        })
    return out


def gen_parity(n, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.randint(1, 99)
        ans = 1 if x % 2 == 0 else 0
        out.append({
            "prompt": f"Q: Is {x} even? 1=yes 0=no\nA: ",
            "answer": str(ans),
            "meta": {"task": "parity", "x": x}
        })
    return out


TASK_REGISTRY = {
    "add_small": gen_add_small,
    "sub_small": gen_sub_small,
    "compare": gen_compare,
    "parity": gen_parity,
}


def tokenize_answer(tokenizer, text):
    ids_bare = tokenizer.encode(text, add_special_tokens=False)
    if len(ids_bare) == 1:
        return ids_bare[0]
    ids_with_space = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids_with_space) == 1:
        return ids_with_space[0]
    return None


@torch.no_grad()
def diagnose_one(model, tokenizer, prompt, answer_token_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]

    probs = F.softmax(last_logits, dim=-1)
    top5_vals, top5_ids = torch.topk(probs, 5)

    predicted_id = last_logits.argmax().item()
    correct = int(predicted_id == answer_token_id)

    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    top2 = torch.topk(last_logits, 2).values
    margin = float((top2[0] - top2[1]).item())
    max_prob = float(probs.max().item())

    return {
        "predicted_id": predicted_id,
        "predicted_text": tokenizer.decode([predicted_id]),
        "correct": correct,
        "entropy": entropy,
        "margin": margin,
        "max_prob": max_prob,
        "top5": [(tokenizer.decode([tid.item()]), f"{prob.item():.4f}", tid.item())
                 for tid, prob in zip(top5_ids, top5_vals)],
        "answer_prob": float(probs[answer_token_id].item()),
    }


def main():
    print("=" * 70)
    print("诊断脚本：验证 prompt/answer/predicted 对齐")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"每个任务打印 {N_DIAG} 个样本\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

    # 先做 tokenizer 基本检查
    print("=" * 70)
    print("Tokenizer 基本检查")
    print("=" * 70)
    for digit in range(10):
        ids_bare = tokenizer.encode(str(digit), add_special_tokens=False)
        ids_space = tokenizer.encode(" " + str(digit), add_special_tokens=False)
        print(f"  '{digit}' -> bare={ids_bare} (n={len(ids_bare)}), "
              f"' {digit}' -> space={ids_space} (n={len(ids_space)})")
    print()

    for task_name, gen_fn in TASK_REGISTRY.items():
        print("=" * 70)
        print(f"Task: {task_name}")
        print("=" * 70)

        samples = gen_fn(N_DIAG, seed=42)

        results = []
        for s in samples:
            tid = tokenize_answer(tokenizer, s["answer"])
            if tid is None:
                print(f"  ⚠️ 答案 '{s['answer']}' 不是单 token，跳过")
                continue
            s["answer_token_id"] = tid
            r = diagnose_one(model, tokenizer, s["prompt"], tid)
            r["prompt"] = s["prompt"]
            r["answer"] = s["answer"]
            r["answer_token_id"] = tid
            r["meta"] = s["meta"]
            results.append(r)

        correct_results = [r for r in results if r["correct"]]
        wrong_results = [r for r in results if not r["correct"]]

        print(f"\n  总样本: {len(results)}, 正确: {len(correct_results)}, "
              f"错误: {len(wrong_results)}")
        print(f"  Accuracy: {len(correct_results)/len(results):.3f}\n")

        # 打印正确样本
        print(f"  --- 正确样本 (前5个) ---")
        for i, r in enumerate(correct_results[:5]):
            print(f"  [{i+1}] Prompt: {r['prompt']!r}")
            print(f"      期望: '{r['answer']}' (id={r['answer_token_id']})")
            print(f"      预测: '{r['predicted_text']}' (id={r['predicted_id']}) ✅")
            print(f"      期望答案概率: {r['answer_prob']:.4f}")
            print(f"      Top-5: {[(t, p) for t, p, _ in r['top5']]}")
            print(f"      entropy={r['entropy']:.3f}, margin={r['margin']:.3f}, "
                  f"max_prob={r['max_prob']:.3f}")
            print()

        # 打印错误样本
        print(f"  --- 错误样本 (前5个) ---")
        for i, r in enumerate(wrong_results[:5]):
            print(f"  [{i+1}] Prompt: {r['prompt']!r}")
            print(f"      期望: '{r['answer']}' (id={r['answer_token_id']})")
            print(f"      预测: '{r['predicted_text']}' (id={r['predicted_id']}) ❌")
            print(f"      期望答案概率: {r['answer_prob']:.4f}")
            print(f"      Top-5: {[(t, p) for t, p, _ in r['top5']]}")
            print(f"      entropy={r['entropy']:.3f}, margin={r['margin']:.3f}, "
                  f"max_prob={r['max_prob']:.3f}")
            print()

        # 信号分布对比
        if correct_results and wrong_results:
            ent_c = np.mean([r["entropy"] for r in correct_results])
            ent_w = np.mean([r["entropy"] for r in wrong_results])
            mar_c = np.mean([r["margin"] for r in correct_results])
            mar_w = np.mean([r["margin"] for r in wrong_results])
            mp_c = np.mean([r["max_prob"] for r in correct_results])
            mp_w = np.mean([r["max_prob"] for r in wrong_results])
            print(f"  --- 信号均值对比 ---")
            print(f"  {'信号':<12} {'正确':>8} {'错误':>8} {'方向':>8}")
            print(f"  {'entropy':<12} {ent_c:>8.3f} {ent_w:>8.3f} "
                  f"{'✅ 正确更低' if ent_c < ent_w else '⚠️ 反向'}")
            print(f"  {'margin':<12} {mar_c:>8.3f} {mar_w:>8.3f} "
                  f"{'✅ 正确更高' if mar_c > mar_w else '⚠️ 反向'}")
            print(f"  {'max_prob':<12} {mp_c:>8.3f} {mp_w:>8.3f} "
                  f"{'✅ 正确更高' if mp_c > mp_w else '⚠️ 反向'}")
            print()

    # compare 任务特别分析
    print("=" * 70)
    print("compare 任务特别分析：答案对齐问题检查")
    print("=" * 70)
    print("compare 的答案是 max(a,b)，即较大数字本身。")
    print("如果 a > b，正确答案是 a；如果 b > a，正确答案是 b。")
    print("模型需要理解'which is larger'并输出较大数字。\n")

    compare_samples = gen_compare(N_DIAG, seed=42)
    for s in compare_samples[:10]:
        a, b = s["meta"]["a"], s["meta"]["b"]
        ans = s["answer"]
        tid = tokenize_answer(tokenizer, ans)
        r = diagnose_one(model, tokenizer, s["prompt"], tid)
        marker = "✅" if r["correct"] else "❌"
        print(f"  Q: Which is larger, {a} or {b}? → 期望={ans}, "
              f"预测='{r['predicted_text']}' {marker}")
        print(f"    Top-3: {[(t, p) for t, p, _ in r['top5'][:3]]}")
    print()

    # parity 任务特别分析
    print("=" * 70)
    print("parity 任务特别分析：答案对齐问题检查")
    print("=" * 70)
    print("parity 的答案是 1(偶数) 或 0(奇数)。")
    print("模型需要理解'Is X even? 1=yes 0=no'并输出 1 或 0。\n")

    parity_samples = gen_parity(N_DIAG, seed=42)
    for s in parity_samples[:10]:
        x = s["meta"]["x"]
        ans = s["answer"]
        tid = tokenize_answer(tokenizer, ans)
        r = diagnose_one(model, tokenizer, s["prompt"], tid)
        marker = "✅" if r["correct"] else "❌"
        print(f"  Q: Is {x} even? → 期望={ans}, "
              f"预测='{r['predicted_text']}' {marker}")
        print(f"    Top-3: {[(t, p) for t, p, _ in r['top5'][:3]]}")
    print()

    print("=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
