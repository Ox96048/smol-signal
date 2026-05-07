import torch
import numpy as np
from typing import Dict, List, Optional
from common.types import SignalOutput, Sample
from common.answer_utils import find_answer_start_position
from signals.registry import compute_signal, list_mmq_signals
from core.auroc import compute_auroc, bootstrap_auroc


def measure_signal(
    signal_type: str,
    logits: torch.Tensor,
    hidden_states: Optional[List[torch.Tensor]] = None,
    **kwargs
) -> Optional[float]:
    if signal_type in ("entropy", "margin"):
        try:
            result = compute_signal(signal_type, logits=logits)
            return result.value
        except Exception:
            return None
    elif signal_type == "self_consistency":
        samples = kwargs.get("samples", None)
        if samples is None:
            return None
        try:
            result = compute_signal("self_consistency", samples=samples)
            return result.value
        except Exception:
            return None
    elif signal_type == "hidden_norm":
        if hidden_states is None:
            return None
        try:
            result = compute_signal("hidden_norm", hidden_states=hidden_states)
            return result.value
        except Exception:
            return None
    return None


def collect_signals(
    sample: Sample,
    model_output: Dict,
    tokenizer=None,
    answer_separator: str = "The answer is",
) -> Dict[str, float]:
    """从模型输出中提取所有信号值.

    关键修正 (v5.1):
      - entropy 只取答案 token 位置的值, 不取全序列均值
      - 如果无法定位答案 token, 回退到最后一个 token
    """
    signal_values = {}
    logits = model_output.get("logits")
    hidden_states = model_output.get("hidden_states")
    input_ids = model_output.get("input_ids")

    if logits is None:
        return signal_values

    if logits.dim() == 3:
        logits = logits.squeeze(0)

    if input_ids is not None and tokenizer is not None:
        answer_pos = find_answer_start_position(
            input_ids, tokenizer, sample.prompt, answer_separator
        )
        if answer_pos < logits.shape[0]:
            answer_logits = logits[answer_pos:]
        else:
            answer_logits = logits[-1:]
    else:
        answer_logits = logits[-1:]

    for signal_type in list_mmq_signals():
        if signal_type in ("entropy", "margin"):
            try:
                result = compute_signal(signal_type, logits=answer_logits)
                signal_values[signal_type] = result.value
            except Exception:
                pass

    if hidden_states is not None:
        try:
            result = compute_signal("hidden_norm", hidden_states=hidden_states)
            signal_values["hidden_norm"] = result.value
        except Exception:
            pass

    return signal_values


def compute_mmq(
    signals: np.ndarray,
    labels: np.ndarray,
) -> float:
    """MMQ = AUROC(signal, correctness). 就这么简单."""
    return compute_auroc(signals, labels)


def compute_mmq_with_ci(
    signals: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """计算 MMQ + bootstrap CI + 统计显著性."""
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
    answer_separator: str = "The answer is",
) -> List[Dict]:
    results = []
    for sample, model_output in zip(samples, model_outputs):
        signal_values = collect_signals(sample, model_output, tokenizer, answer_separator)
        results.append({
            "sample_id": sample.meta.get("id", "unknown") if sample.meta else "unknown",
            "signals": signal_values,
        })
    return results
