import pytest
import torch
import math
from signals.entropy import compute_entropy, compute_entropy_signal


def test_uniform_distribution_entropy():
    logits = torch.zeros(1, 10)
    entropy = compute_entropy(logits)
    expected = math.log(10)
    assert abs(entropy - expected) < 0.01, f"Expected {expected}, got {entropy}"


def test_deterministic_distribution_entropy():
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    entropy = compute_entropy(logits)
    assert entropy < 0.01, f"Expected near 0, got {entropy}"


def test_binary_uniform_entropy():
    logits = torch.zeros(1, 2)
    entropy = compute_entropy(logits)
    expected = math.log(2)
    assert abs(entropy - expected) < 0.01, f"Expected {expected}, got {entropy}"


def test_entropy_signal_output():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    output = compute_entropy_signal(logits)
    assert hasattr(output, 'value')
    assert hasattr(output, 'confidence')
    assert hasattr(output, 'metadata')
    assert output.confidence == 1.0
    assert output.metadata['method'] == 'softmax_entropy'


def test_entropy_with_batch_input():
    logits = torch.zeros(3, 5)
    entropy = compute_entropy(logits)
    expected = math.log(5)
    assert abs(entropy - expected) < 0.01, f"Expected {expected}, got {entropy}"


def test_entropy_range():
    logits = torch.randn(1, 100)
    entropy = compute_entropy(logits)
    assert entropy >= 0, f"Entropy cannot be negative, got {entropy}"
    assert entropy <= math.log(100), f"Entropy exceeds maximum, got {entropy}"