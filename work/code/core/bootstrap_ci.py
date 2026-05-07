import numpy as np
from typing import List, Tuple, Optional, Callable


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Optional[Callable] = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if stat_func is None:
        stat_func = np.mean
    
    n_samples = len(data)
    boot_statistics = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = data[indices]
        stat = stat_func(bootstrap_sample)
        boot_statistics.append(stat)
    
    boot_statistics = np.array(boot_statistics)
    lower_pct = (1 - confidence_level) / 2 * 100
    upper_pct = (1 + confidence_level) / 2 * 100
    
    lower = np.percentile(boot_statistics, lower_pct)
    upper = np.percentile(boot_statistics, upper_pct)
    
    return lower, upper


def compute_mmq_ci(
    mmq_scores: List[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    data = np.array(mmq_scores)
    return bootstrap_ci(data, stat_func=np.mean, n_bootstraps=n_bootstraps,
                       confidence_level=confidence_level, random_seed=random_seed)


def compute_auroc_ci(
    auroc_scores: List[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    data = np.array(auroc_scores)
    return bootstrap_ci(data, stat_func=np.mean, n_bootstraps=n_bootstraps,
                       confidence_level=confidence_level, random_seed=random_seed)


def bootstrap_statistics(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: Optional[int] = None
) -> dict:
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(data)
    boot_means = []
    boot_stds = []
    boot_medians = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = data[indices]
        boot_means.append(np.mean(bootstrap_sample))
        boot_stds.append(np.std(bootstrap_sample))
        boot_medians.append(np.median(bootstrap_sample))
    
    return {
        "mean": np.mean(boot_means),
        "std": np.mean(boot_stds),
        "median": np.mean(boot_medians),
        "mean_ci": bootstrap_ci(data, stat_func=np.mean, n_bootstraps=n_bootstraps),
        "median_ci": bootstrap_ci(data, stat_func=np.median, n_bootstraps=n_bootstraps)
    }