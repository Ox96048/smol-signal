import pytest
import torch
from signals.max_logit import compute_max_logit, compute_max_logit_signal


def test_max_logit_basic():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    max_val = compute_max_logit(logits)
    assert abs(max_val - 3.0) < 0.01, f"Expected 3.0, got {max_val}"


def test_max_logit_with_negative_values():
    logits = torch.tensor([[-5.0, -2.0, -8.0]])
    max_val = compute_max_logit(logits)
    assert abs(max_val - (-2.0)) < 0.01, f"Expected -2.0, got {max_val}"


def test_max_logit_with_zero():
    logits = torch.tensor([[0.0, -1.0, 1.0]])
    max_val = compute_max_logit(logits)
    assert abs(max_val - 1.0) < 0.01, f"Expected 1.0, got {max_val}"


def test_max_logit_signal_output():
    logits = torch.tensor([[5.0, 2.0, 8.0]])
    output = compute_max_logit_signal(logits)
    assert hasattr(output, 'value')
    assert hasattr(output, 'confidence')
    assert hasattr(output, 'metadata')
    assert output.confidence == 1.0
    assert output.metadata['method'] == 'max_logit'


def test_max_logit_with_batch_input():
    logits = torch.tensor([[10.0, 5.0], [8.0, 12.0]])
    max_val = compute_max_logit(logits)
    expected = (10.0 + 12.0) / 2.0
    assert abs(max_val - expected) < 0.01, f"Expected {expected}, got {max_val}"


def test_max_logit_returns_float():
    logits = torch.tensor([[3.0, 1.0]])
    result = compute_max_logit(logits)
    assert isinstance(result, float), f"Expected float, got {type(result)}"