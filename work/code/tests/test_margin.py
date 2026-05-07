import pytest
import torch
from signals.margin import compute_margin, compute_margin_signal


def test_margin_with_clear_winner():
    logits = torch.tensor([[10.0, 5.0, 1.0]])
    margin = compute_margin(logits)
    assert abs(margin - 5.0) < 0.01, f"Expected 5.0, got {margin}"


def test_margin_with_tie():
    logits = torch.tensor([[5.0, 5.0, 1.0]])
    margin = compute_margin(logits)
    assert abs(margin - 0.0) < 0.01, f"Expected 0.0, got {margin}"


def test_margin_with_negative_logits():
    logits = torch.tensor([[-1.0, -2.0, -5.0]])
    margin = compute_margin(logits)
    assert abs(margin - 1.0) < 0.01, f"Expected 1.0, got {margin}"


def test_margin_signal_output():
    logits = torch.tensor([[3.0, 1.0, 0.5]])
    output = compute_margin_signal(logits)
    assert hasattr(output, 'value')
    assert hasattr(output, 'confidence')
    assert hasattr(output, 'metadata')
    assert output.confidence == 1.0
    assert output.metadata['method'] == 'top2_margin'


def test_margin_with_batch_input():
    logits = torch.tensor([[10.0, 5.0], [8.0, 2.0]])
    margin = compute_margin(logits)
    expected = (5.0 + 6.0) / 2.0
    assert abs(margin - expected) < 0.01, f"Expected {expected}, got {margin}"


def test_margin_order_independence():
    logits1 = torch.tensor([[10.0, 3.0]])
    logits2 = torch.tensor([[3.0, 10.0]])
    margin1 = compute_margin(logits1)
    margin2 = compute_margin(logits2)
    assert abs(margin1 - margin2) < 0.01, f"Margin should be order-independent"