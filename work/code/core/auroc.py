"""AUROC 计算与 Bootstrap 置信区间.

这是 Phase 0 的核心判决工具。
每个 MMQ(signal, D) = AUROC(signal_values, correctness_labels)
Bootstrap CI 判定信号是否显著优于随机 (AUROC > 0.5).
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score


def compute_auroc(
    signals: np.ndarray,
    labels: np.ndarray,
    pos_label: int = 1,
) -> float:
    """计算 AUROC.

    参数:
      signals: [N] 信号值 (entropy/margin/...)
      labels: [N] 正确性标签 (1=正确, 0=错误)
      pos_label: 正类标签, 通常 1

    返回: AUROC ∈ [0.5, 1.0]

    注意: sklearn 的 AUROC 假设"信号值越大→越可能是正类"。
    对于 entropy (高熵=不确定=易错), 是反向关系。
    调用方需要对 '反向信号' 做 1 - signal 变换后再传入。
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    if len(signals) < 2:
        return 0.5

    try:
        return float(roc_auc_score(labels, signals))
    except ValueError:
        return 0.5


def bootstrap_auroc(
    signals: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap AUROC 估计: (mean, lower_ci, upper_ci).

    返回:
      mean: bootstrap AUROC 均值
      lower_ci: 2.5% 分位数
      upper_ci: 97.5% 分位数

    CI 宽度 = upper - lower.
    如果 CI 宽度 > 0.05, Phase 0 GATE 1 不通过 (仪器不稳健).
    如果 CI 包含 0.5, 信号不显著.
    """
    rng = np.random.RandomState(seed)
    n = len(signals)
    auroc_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_signals = signals[idx]
        boot_labels = labels[idx]
        auroc_samples[i] = compute_auroc(boot_signals, boot_labels)

    mean = float(np.mean(auroc_samples))
    lower = float(np.percentile(auroc_samples, 2.5))
    upper = float(np.percentile(auroc_samples, 97.5))

    return mean, lower, upper


def delong_test(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """DeLong test: AUROC(a) vs AUROC(b) 的显著性检验.

    用于判决 ΔMMQ(a, b) > 0 是否统计显著.

    返回: (p_value, statistic)
      p_value < 0.05 → a 显著优于 b
    """
    from scipy.stats import norm

    n = len(labels)
    if n < 10:
        return 1.0, 0.0

    auroc_a = compute_auroc(signals_a, labels)
    auroc_b = compute_auroc(signals_b, labels)

    # DeLong variance estimate (简化版, 适用于独立检验)
    diff = auroc_a - auroc_b
    if abs(diff) < 1e-8:
        return 1.0, 0.0

    # Jackknife variance
    jackknife_diffs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        a_jk = compute_auroc(signals_a[mask], labels[mask])
        b_jk = compute_auroc(signals_b[mask], labels[mask])
        jackknife_diffs[i] = a_jk - b_jk

    se = np.std(jackknife_diffs) * np.sqrt(n - 1)
    if se < 1e-10:
        return 1.0, 0.0

    z = diff / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))

    return float(p), float(z)


def multiple_comparison_correction(
    p_values: list,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> list:
    """多重比较校正.

    补丁建议 §7: 主判决用 Bonferroni, 次判决用 BH FDR.

    参数:
      p_values: 原始 p 值列表
      method: "bonferroni" | "bh" (Benjamini-Hochberg)
      alpha: 显著性水平

    返回: 校正后的 p 值列表
    """
    import numpy as np
    p = np.array(p_values)
    n = len(p)

    if method == "bonferroni":
        corrected = np.minimum(p * n, 1.0)
        rejected = corrected <= alpha
        return corrected.tolist()

    elif method == "bh":
        order = np.argsort(p)
        ranks = np.arange(1, n + 1)
        bh_threshold = alpha * ranks / n
        corrected = p[order] * n / ranks
        corrected = np.minimum(1.0, np.maximum.accumulate(corrected))
        result = np.zeros(n)
        result[order] = corrected
        return result.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")
