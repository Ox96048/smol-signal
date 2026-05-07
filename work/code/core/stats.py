"""Bootstrap AUROC CI。只用 numpy + sklearn，不依赖其它。"""
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple


def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    scores: [N] 预测分数（比如 entropy 的负数，高分=高置信）
    labels: [N] 0/1 正确性标签（1=正确）

    注意信号方向：entropy 越高意味着越不确定，所以正确性标签和 entropy
    应该负相关。如果算出的 AUROC < 0.5，代表信号有用但方向反了。
    统一做法：AUROC 自动取 max(auroc, 1-auroc) 是错的——会丢信息。
    这里返回原始 AUROC 和 direction 说明。
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
