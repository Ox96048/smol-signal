"""
合成任务。每个任务函数返回 list of dict:
  {"prompt": str, "answer": str, "meta": dict}

注意：所有答案必须是 tokenizer_recon 确认过的"单 token 答案"。
先跑 tokenizer_recon.py 再跑这个。
"""
import random
from typing import List, Dict


def gen_add_small(n: int, seed: int = 42) -> List[Dict]:
    """个位数加法，和 <= 9。答案保证是单 token 数字。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9 - a)
        ans = a + b
        out.append({
            "prompt": f"Q: What is {a} + {b}?\nA:",
            "answer": str(ans),
            "meta": {"task": "add_small", "a": a, "b": b, "difficulty": "easy"}
        })
    return out


def gen_sub_small(n: int, seed: int = 42) -> List[Dict]:
    """个位数减法，结果 >= 0 且 <= 9。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, a)
        ans = a - b
        out.append({
            "prompt": f"Q: What is {a} - {b}?\nA:",
            "answer": str(ans),
            "meta": {"task": "sub_small", "a": a, "b": b, "difficulty": "easy"}
        })
    return out


def gen_compare(n: int, seed: int = 42) -> List[Dict]:
    """两数比较：大的回答 A，小的回答 B。答案是 A/B。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        while a == b:
            b = rng.randint(1, 99)
        ans = "A" if a > b else "B"
        out.append({
            "prompt": f"Q: Which is larger, A={a} or B={b}?\nA:",
            "answer": ans,
            "meta": {"task": "compare", "a": a, "b": b, "difficulty": "medium"}
        })
    return out


def gen_yes_no_parity(n: int, seed: int = 42) -> List[Dict]:
    """数字奇偶判断：yes=偶数，no=奇数。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.randint(1, 99)
        ans = "yes" if x % 2 == 0 else "no"
        out.append({
            "prompt": f"Q: Is {x} an even number?\nA:",
            "answer": ans,
            "meta": {"task": "parity", "x": x, "difficulty": "medium"}
        })
    return out


TASK_REGISTRY = {
    "add_small": gen_add_small,
    "sub_small": gen_sub_small,
    "compare": gen_compare,
    "parity": gen_yes_no_parity,
}
