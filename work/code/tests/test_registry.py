import pytest
import torch
from signals.registry import (
    get_signal_fn,
    register_signal,
    list_signals,
    compute_signal,
    SIGNAL_REGISTRY
)


def test_register_signal():
    def test_signal(**kwargs):
        return {"value": 0.5}
    
    register_signal("test_signal", test_signal)
    assert "test_signal" in SIGNAL_REGISTRY
    assert SIGNAL_REGISTRY["test_signal"] == test_signal


def test_get_signal_fn():
    fn = get_signal_fn("entropy")
    assert fn is not None
    assert callable(fn)


def test_get_signal_fn_unknown():
    fn = get_signal_fn("unknown_signal")
    assert fn is None


def test_list_signals():
    signals = list_signals()
    assert isinstance(signals, list)
    assert len(signals) > 0
    assert "entropy" in signals
    assert "margin" in signals
    assert "max_logit" in signals


def test_compute_signal_entropy():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    result = compute_signal("entropy", logits=logits)
    assert hasattr(result, 'value')
    assert result.value > 0


def test_compute_signal_margin():
    logits = torch.tensor([[10.0, 5.0, 1.0]])
    result = compute_signal("margin", logits=logits)
    assert hasattr(result, 'value')
    assert result.value > 0


def test_compute_signal_max_logit():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    result = compute_signal("max_logit", logits=logits)
    assert hasattr(result, 'value')
    assert result.value == 5.0


def test_compute_signal_hidden_norm():
    hidden_states = [torch.tensor([[[1.0, 2.0, 3.0]]])]
    result = compute_signal("hidden_norm", hidden_states=hidden_states)
    assert hasattr(result, 'value')
    assert result.value > 0


def test_compute_signal_self_consistency():
    samples = ["A", "A", "B"]
    result = compute_signal("self_consistency", samples=samples)
    assert hasattr(result, 'value')
    assert 0 <= result.value <= 1


def test_compute_signal_unknown():
    with pytest.raises(ValueError):
        compute_signal("unknown", logits=torch.tensor([[1.0, 2.0]]))