import pytest
import numpy as np
from core.bootstrap_ci import bootstrap_ci, compute_mmq_ci, compute_auroc_ci, bootstrap_statistics


def test_bootstrap_ci_normal_distribution():
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    
    lower, upper = bootstrap_ci(data, n_bootstraps=1000, random_seed=42)
    
    true_mean = 100
    assert lower < true_mean < upper, f"CI ({lower}, {upper}) does not contain true mean {true_mean}"


def test_bootstrap_ci_uniform_distribution():
    np.random.seed(42)
    data = np.random.uniform(0, 1, 500)
    
    lower, upper = bootstrap_ci(data, stat_func=np.mean, n_bootstraps=500, random_seed=42)
    
    true_mean = 0.5
    assert lower < true_mean < upper, f"CI ({lower}, {upper}) does not contain true mean {true_mean}"


def test_bootstrap_ci_known_distribution():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    lower, upper = bootstrap_ci(data, stat_func=np.mean, n_bootstraps=1000, random_seed=42)
    
    sample_mean = np.mean(data)
    assert lower <= sample_mean <= upper


def test_compute_mmq_ci():
    mmq_scores = [0.6, 0.7, 0.8, 0.75, 0.65, 0.85, 0.72, 0.68, 0.78, 0.73]
    
    lower, upper = compute_mmq_ci(mmq_scores, n_bootstraps=500, random_seed=42)
    
    sample_mean = np.mean(mmq_scores)
    assert lower <= sample_mean <= upper
    assert lower >= 0
    assert upper <= 1


def test_compute_auroc_ci():
    auroc_scores = [0.75, 0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77]
    
    lower, upper = compute_auroc_ci(auroc_scores, n_bootstraps=500, random_seed=42)
    
    sample_mean = np.mean(auroc_scores)
    assert lower <= sample_mean <= upper
    assert lower >= 0.5
    assert upper <= 1.0


def test_bootstrap_statistics():
    np.random.seed(42)
    data = np.random.normal(50, 10, 200)
    
    stats = bootstrap_statistics(data, n_bootstraps=500, random_seed=42)
    
    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats
    assert "mean_ci" in stats
    assert "median_ci" in stats
    
    assert stats["mean_ci"][0] <= stats["mean"] <= stats["mean_ci"][1]


def test_bootstrap_ci_empty_data():
    data = np.array([])
    with pytest.raises(Exception):
        bootstrap_ci(data)


def test_bootstrap_ci_single_sample():
    data = np.array([42.0])
    lower, upper = bootstrap_ci(data, n_bootstraps=100, random_seed=42)
    assert lower == 42.0
    assert upper == 42.0