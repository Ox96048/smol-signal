"""单元测试。必须全部通过才能继续。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import numpy as np
from signals.entropy import compute_entropy
from signals.margin import compute_margin
from signals.max_logit import compute_max_prob
from signals.signals import compute_neg_log_prob_correct
from core.stats import bootstrap_auroc_ci


def test_entropy_uniform():
    """均匀分布的熵应该是 log(V)"""
    V = 100
    logits = torch.zeros(V)
    e = compute_entropy(logits)
    expected = math.log(V)
    assert abs(e - expected) < 1e-5, f"expected {expected}, got {e}"
    print(f"✅ test_entropy_uniform: H={e:.4f}, expected={expected:.4f}")


def test_entropy_peaked():
    """极端尖峰分布熵应接近 0"""
    logits = torch.zeros(100)
    logits[0] = 100.0
    e = compute_entropy(logits)
    assert e < 1e-3, f"expected near 0, got {e}"
    print(f"✅ test_entropy_peaked: H={e:.6f}")


def test_entropy_two_peaks():
    """两个相等的峰：熵应该是 log(2)"""
    logits = torch.full((100,), -1000.0)
    logits[0] = 0.0
    logits[1] = 0.0
    e = compute_entropy(logits)
    expected = math.log(2)
    assert abs(e - expected) < 1e-4, f"expected {expected}, got {e}"
    print(f"✅ test_entropy_two_peaks: H={e:.4f}, expected={expected:.4f}")


def test_entropy_1d_assert():
    """2D logits 应该触发 assert"""
    logits_2d = torch.zeros(1, 100)
    try:
        compute_entropy(logits_2d)
        assert False, "should have raised AssertionError"
    except AssertionError:
        print("✅ test_entropy_1d_assert: 2D logits correctly rejected")


def test_margin():
    """margin 测试：top1=10, top2=3, 差=7"""
    logits = torch.tensor([10.0, 3.0, 1.0, -1.0, -5.0])
    m = compute_margin(logits)
    assert abs(m - 7.0) < 1e-5
    print(f"✅ test_margin: {m}")


def test_max_prob():
    """单一尖峰 max_prob 接近 1"""
    logits = torch.full((100,), -100.0)
    logits[0] = 100.0
    p = compute_max_prob(logits)
    assert p > 0.99
    print(f"✅ test_max_prob: {p:.6f}")


def test_max_prob_range():
    """max_prob 应该在 [0, 1]"""
    logits = torch.randn(1000)
    p = compute_max_prob(logits)
    assert 0.0 <= p <= 1.0, f"max_prob out of range: {p}"
    print(f"✅ test_max_prob_range: {p:.4f} ∈ [0, 1]")


def test_neg_log_prob_correct():
    """oracle 信号：正确 token 的负对数概率"""
    logits = torch.zeros(10)
    logits[3] = 5.0
    nlp = compute_neg_log_prob_correct(logits, 3)
    assert nlp > 0, f"neg_log_prob should be positive, got {nlp}"
    assert nlp < 5.0, f"neg_log_prob too large for peaked distribution: {nlp}"
    print(f"✅ test_neg_log_prob_correct: {nlp:.4f}")


def test_bootstrap_perfect_signal():
    """完美可分信号：AUROC 应接近 1"""
    np.random.seed(42)
    scores = np.concatenate([np.random.randn(100) + 5, np.random.randn(100)])
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    r = bootstrap_auroc_ci(scores, labels, n_bootstrap=500)
    assert r["auroc_mean"] > 0.95, r
    assert r["ci_width"] < 0.1, r
    print(f"✅ test_bootstrap_perfect: AUROC={r['auroc_mean']:.3f}, "
          f"CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


def test_bootstrap_random_signal():
    """随机信号：AUROC 应接近 0.5"""
    np.random.seed(0)
    scores = np.random.randn(500)
    labels = (np.random.rand(500) > 0.5).astype(int)
    r = bootstrap_auroc_ci(scores, labels, n_bootstrap=500)
    assert 0.4 < r["auroc_mean"] < 0.6, r
    print(f"✅ test_bootstrap_random: AUROC={r['auroc_mean']:.3f}, "
          f"CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


def test_bootstrap_degenerate():
    """全是同一类：应返回 degenerate"""
    np.random.seed(0)
    scores = np.random.randn(100)
    labels = np.ones(100)
    r = bootstrap_auroc_ci(scores, labels)
    assert r["degenerate"] == True
    print(f"✅ test_bootstrap_degenerate: {r['reason']}")


if __name__ == "__main__":
    test_entropy_uniform()
    test_entropy_peaked()
    test_entropy_two_peaks()
    test_entropy_1d_assert()
    test_margin()
    test_max_prob()
    test_max_prob_range()
    test_neg_log_prob_correct()
    test_bootstrap_perfect_signal()
    test_bootstrap_random_signal()
    test_bootstrap_degenerate()
    print("\n🎉 所有测试通过")
