"""H0-1: Tokenizer 侦查脚本。

运行方式: python work/scripts/tokenizer_investigate.py

目的:
  1. 确认 SmolLM-135M tokenizer 对各种答案格式的 tokenization 行为
  2. 确定哪些任务类型答案 = 单 token（低复杂度），哪些 = 多 token
  3. 输出直接决定任务选择和 answer 提取策略
"""

from transformers import AutoTokenizer


def investigate():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("=" * 60)
    print("Tokenizer: HuggingFaceTB/SmolLM-135M (base, NOT instruct)")
    print(f"Vocab size: {tok.vocab_size}")
    print(f"BOS token: {tok.bos_token!r} -> {tok.encode(tok.bos_token or '', add_special_tokens=False)}")
    print(f"EOS token: {tok.eos_token!r} -> {tok.encode(tok.eos_token or '', add_special_tokens=False)}")
    print("=" * 60)

    candidates = [
        # 数字
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "42", "100", "99", "256", "999", "1000",
        # 单词
        "yes", "no", "Yes", "No", "YES", "NO",
        "true", "false", "True", "False", "TRUE", "FALSE",
        "larger", "smaller", "equal", "Larger", "Smaller",
        "A", "B", "C", "D",
        # 多词
        "Alice", "Bob", "Charlie",
        "5", "five",
    ]

    print(f"\n{'Answer':<12} {'Token IDs':<30} {'N tokens':>8} {'Decode back':<16}")
    print("-" * 70)

    results = {}
    for ans in candidates:
        ids = tok.encode(ans, add_special_tokens=False)
        decoded = tok.decode(ids)
        results[ans] = {"ids": ids, "n_tokens": len(ids), "decoded": decoded}
        ids_str = str(ids)
        print(f"{ans!r:<12} {ids_str:<30} {len(ids):>8} {decoded!r:<16}")

    print("\n" + "=" * 60)
    print("关键发现:")
    print("=" * 60)

    single_token = [(k, v) for k, v in results.items() if v["n_tokens"] == 1]
    multi_token = [(k, v) for k, v in results.items() if v["n_tokens"] > 1]

    print(f"\n单 token 答案 ({len(single_token)} 个):")
    for ans, info in single_token:
        print(f"  {ans!r} -> id={info['ids'][0]}")

    print(f"\n多 token 答案 ({len(multi_token)} 个):")
    for ans, info in multi_token:
        print(f"  {ans!r} -> {info['ids']} ({info['n_tokens']} tokens)")

    print("\n" + "=" * 60)
    print("任务选择建议:")
    print("=" * 60)
    print("✅ 优先: 答案为单 token 的任务（entropy 信号干净）")
    print("   例: yes/no 判断、选项 ABCD")
    print("✅ 可行: 两位数字（2 tokens），但 SPL target 需处理")
    print("   建议: target = 末位 token loss（补丁清单 §2 提醒）")
    print("⚠️  慎重: 三位以上数字或长答案，token 数 > 3")
    print("   熵信号被稀释，SPL target 噪声大")

    # 检查 prompt 模板的 answer token 位置
    print("\n" + "=" * 60)
    print("Prompt 模板检查:")
    print("=" * 60)
    templates = [
        "What is 12 + 7? The answer is",
        "Q: Alice has 3 apples, Bob has 5. How many in total?\nA:",
        "Is 15 larger than 22? Answer with yes or no:",
    ]
    for t in templates:
        ids = tok.encode(t, add_special_tokens=False)
        last_few = tok.decode(ids[-5:])
        print(f"  Prompt: {t[:50]}...")
        print(f"    Last 5 tokens: {ids[-5:]} -> {last_few!r}")
        print(f"    Total tokens: {len(ids)}")
        print()


if __name__ == "__main__":
    investigate()
