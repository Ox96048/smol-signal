import pytest
import torch
from signals.hidden_norm import compute_hidden_norm, compute_hidden_norm_signal


def test_hidden_norm_basic():
    hidden_states = [
        torch.randn(1, 5, 64),
        torch.randn(1, 5, 64),
        torch.tensor([[[1.0, 2.0, 3.0]]])
    ]
    norm = compute_hidden_norm(hidden_states)
    expected = torch.norm(torch.tensor([1.0, 2.0, 3.0])).item()
    assert abs(norm - expected) < 0.01, f"Expected {expected}, got {norm}"


def test_hidden_norm_identity():
    hidden_states = [torch.tensor([[[1.0, 0.0, 0.0]]])]
    norm = compute_hidden_norm(hidden_states)
    assert abs(norm - 1.0) < 0.01, f"Expected 1.0, got {norm}"


def test_hidden_norm_zero_vector():
    hidden_states = [torch.tensor([[[0.0, 0.0, 0.0]]])]
    norm = compute_hidden_norm(hidden_states)
    assert abs(norm - 0.0) < 0.01, f"Expected 0.0, got {norm}"


def test_hidden_norm_l2():
    hidden_states = [torch.tensor([[[3.0, 4.0]]])]
    norm = compute_hidden_norm(hidden_states, p=2)
    assert abs(norm - 5.0) < 0.01, f"Expected 5.0, got {norm}"


def test_hidden_norm_l1():
    hidden_states = [torch.tensor([[[3.0, 4.0]]])]
    norm = compute_hidden_norm(hidden_states, p=1)
    assert abs(norm - 7.0) < 0.01, f"Expected 7.0, got {norm}"


def test_hidden_norm_signal_output():
    hidden_states = [torch.tensor([[[1.0, 2.0, 3.0]]])]
    output = compute_hidden_norm_signal(hidden_states)
    assert hasattr(output, 'value')
    assert hasattr(output, 'confidence')
    assert hasattr(output, 'metadata')
    assert output.confidence == 1.0
    assert output.metadata['method'] == 'hidden_norm'
    assert output.metadata['norm_type'] == 2


def test_hidden_norm_batch():
    hidden_states = [
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    ]
    norm = compute_hidden_norm(hidden_states)
    expected = (1.0 + 1.0) / 2.0
    assert abs(norm - expected) < 0.01, f"Expected {expected}, got {norm}"