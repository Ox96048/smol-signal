"""self_consistency 信号：多次采样一致性"""
from typing import List


def compute_self_consistency(
    samples: List[str],
    metric: str = "exact_match"
) -> float:
    """
    samples: 同一 prompt 多次采样的输出列表
    metric: "exact_match" 或 "edit_distance"
    returns: 0-1 之间的一致性分数
    """
    if len(samples) == 0:
        return 0.0
    if len(samples) == 1:
        return 1.0

    matches = 0
    total = len(samples) * (len(samples) - 1) / 2

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if metric == "exact_match":
                if samples[i] == samples[j]:
                    matches += 1
            elif metric == "edit_distance":
                dist = sum(c1 != c2 for c1, c2 in zip(samples[i], samples[j]))
                max_len = max(len(samples[i]), len(samples[j]))
                if max_len > 0 and (max_len - dist) / max_len > 0.9:
                    matches += 1

    return matches / total if total > 0 else 0.0


def compute_self_consistency_signal(samples: List[str], **kwargs) -> float:
    return compute_self_consistency(samples)
