import pytest
from signals.self_consistency import compute_self_consistency, compute_self_consistency_signal


def test_self_consistency_all_same():
    samples = ["A", "A", "A", "A"]
    consistency = compute_self_consistency(samples)
    assert abs(consistency - 1.0) < 0.01, f"Expected 1.0, got {consistency}"


def test_self_consistency_all_different():
    samples = ["A", "B", "C", "D"]
    consistency = compute_self_consistency(samples)
    assert abs(consistency - 0.0) < 0.01, f"Expected 0.0, got {consistency}"


def test_self_consistency_mixed():
    samples = ["A", "A", "B", "B"]
    consistency = compute_self_consistency(samples)
    expected = 2 / 6  # AA, BB matches out of 6 pairs
    assert abs(consistency - expected) < 0.01, f"Expected {expected}, got {consistency}"


def test_self_consistency_single_sample():
    samples = ["A"]
    consistency = compute_self_consistency(samples)
    assert abs(consistency - 1.0) < 0.01, f"Expected 1.0, got {consistency}"


def test_self_consistency_empty():
    samples = []
    consistency = compute_self_consistency(samples)
    assert abs(consistency - 0.0) < 0.01, f"Expected 0.0, got {consistency}"


def test_self_consistency_two_samples_same():
    samples = ["Yes", "Yes"]
    consistency = compute_self_consistency(samples)
    assert abs(consistency - 1.0) < 0.01, f"Expected 1.0, got {consistency}"


def test_self_consistency_two_samples_different():
    samples = ["Yes", "No"]
    consistency = compute_self_consistency(samples)
    assert abs(consistency - 0.0) < 0.01, f"Expected 0.0, got {consistency}"


def test_self_consistency_signal_output():
    samples = ["A", "A", "B"]
    output = compute_self_consistency_signal(samples)
    assert hasattr(output, 'value')
    assert hasattr(output, 'confidence')
    assert hasattr(output, 'metadata')
    assert output.confidence == 1.0
    assert output.metadata['method'] == 'self_consistency'
    assert output.metadata['num_samples'] == 3


def test_self_consistency_edit_distance():
    samples = ["Hello", "Hello", "Helo", "Hello"]
    consistency = compute_self_consistency(samples, metric="edit_distance")
    assert consistency >= 0.5, f"Expected >= 0.5, got {consistency}"