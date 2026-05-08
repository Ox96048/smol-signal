import pytest
import torch
import numpy as np
from common.types import Sample
from core.measure_mmq import collect_signals, compute_mmq, compute_mmq_with_ci, batch_measure


def test_collect_signals_basic():
    sample = Sample(
        prompt="2 + 2 = ?",
        answer="4",
        meta={"id": "test_001"}
    )

    model_output = {
        "logits": torch.tensor([[10.0, -1.0, -1.0]]),
        "hidden_states": [torch.tensor([[[1.0, 2.0, 3.0]]])],
        "input_ids": torch.tensor([[1, 2, 3]])
    }

    result = collect_signals(sample, model_output)

    assert "entropy" in result
    assert "margin" in result


def test_compute_mmq_basic():
    signals = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    auroc = compute_mmq(signals, labels)
    assert auroc == 1.0


def test_compute_mmq_random():
    signals = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    auroc = compute_mmq(signals, labels)
    assert auroc == 0.5


def test_compute_mmq_with_ci():
    np.random.seed(42)
    n = 200
    signals = np.random.randn(n)
    labels = (signals > 0).astype(int)
    signals_noisy = signals + np.random.randn(n) * 0.5

    result = compute_mmq_with_ci(signals_noisy, labels, n_bootstrap=500, seed=42)

    assert "auroc" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "ci_width" in result
    assert result["ci_width"] > 0
    assert 0.5 <= result["auroc"] <= 1.0


def test_batch_measure():
    samples = [
        Sample(prompt="Q1", answer="A1", meta={"id": "s1"}),
        Sample(prompt="Q2", answer="A2", meta={"id": "s2"})
    ]

    model_outputs = [
        {"logits": torch.tensor([[10.0, 1.0]]), "input_ids": torch.tensor([[1, 2]])},
        {"logits": torch.tensor([[5.0, 5.0]]), "input_ids": torch.tensor([[1, 2]])}
    ]

    results = batch_measure(samples, model_outputs)

    assert len(results) == 2
    assert results[0]["sample_id"] == "s1"
    assert results[1]["sample_id"] == "s2"
