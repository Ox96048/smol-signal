"""
H0-1: Tokenizer 侦查
目的：回答"SmolLM-135M 的答案 token 怎么切"
输出：results/tokenizer_recon.json
"""
import json
import os
from transformers import AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM-135M"


def run_recon(output_dir: str = "results"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    candidates = {
        "single_digit": [str(i) for i in range(0, 10)],
        "two_digit": [str(i) for i in [10, 12, 19, 42, 77, 99]],
        "word_answers": ["yes", "no", "A", "B", "C", "true", "false",
                         "larger", "smaller", "equal"],
        "with_leading_space": [f" {i}" for i in range(0, 10)],
    }

    results = {"model": MODEL_ID, "categories": {}}

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

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tokenizer_recon.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    r = run_recon()
    print(f"单 token 答案数: {r['single_token_count']}")
    print(f"\n前 20 个单 token 答案:")
    for s in r['single_token_answers'][:20]:
        print(f"  '{s['text']}' -> id={s['token_id']} (类别={s['category']})")
    print(f"\nprompt 尾部分析:")
    for p in r['prompt_tail_analysis']:
        print(f"  '{p['prompt']}' -> 尾5token: {p['last_5_tokens']}")
