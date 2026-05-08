import torch
import numpy as np
from typing import Dict, List, Optional
from common.types import Sample
from signals.registry import compute_signal, list_signals
from core.auroc import compute_auroc, bootstrap_auroc


def measure_signal(
    signal_type: str,
    logits: torch.Tensor,
    **kwargs
) -> Optional[float]:
    try:
        return compute_signal(signal_type, logits=logits, **kwargs)
    except Exception:
        return None


def collect_signals(
    sample: Sample,
    model_output: Dict,
    tokenizer=None,
) -> Dict[str, float]:
    signal_values = {}
    logits = model_output.get("logits")

    if logits is None:
        return signal_values

    if logits.dim() == 3:
        logits = logits.squeeze(0)

    answer_logits = logits[-1:]

    for signal_type in list_signals():
        if signal_type in ("entropy", "margin", "max_prob"):
            try:
                val = compute_signal(signal_type, logits=answer_logits)
                signal_values[signal_type] = float(val)
            except Exception:
                pass

    return signal_values


def compute_mmq(
    signals: np.ndarray,
    labels: np.ndarray,
) -> float:
    return compute_auroc(signals, labels)


def compute_mmq_with_ci(
    signals: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    point_estimate = compute_auroc(signals, labels)
    mean, lower, upper = bootstrap_auroc(signals, labels, n_bootstrap, seed)
    ci_width = upper - lower
    significant = lower > 0.5

    return {
        "auroc": point_estimate,
        "bootstrap_mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": ci_width,
        "significant": significant,
        "gate1_pass": ci_width <= 0.05,
    }


def batch_measure(
    samples: List[Sample],
    model_outputs: List[Dict],
    tokenizer=None,
) -> List[Dict]:
    results = []
    for sample, model_output in zip(samples, model_outputs):
        signal_values = collect_signals(sample, model_output, tokenizer)
        results.append({
            "sample_id": sample.meta.get("id", "unknown") if sample.meta else "unknown",
            "signals": signal_values,
        })
    return results
