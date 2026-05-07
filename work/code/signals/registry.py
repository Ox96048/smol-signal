"""信号注册中心：统一入口，signal_type 字符串 -> 函数"""
from typing import Dict, Callable, Optional, Any

from .entropy import compute_entropy, compute_entropy_signal
from .margin import compute_margin, compute_margin_signal
from .max_logit import compute_max_prob, compute_max_logit_signal
from .hidden_norm import compute_hidden_norm, compute_hidden_norm_signal
from .self_consistency import compute_self_consistency, compute_self_consistency_signal
from .verbalized import compute_verbalized_signal, build_prompt
from .signals import compute_neg_log_prob_correct, SIGNAL_REGISTRY


SIGNAL_REGISTRY_FULL: Dict[str, Callable] = {}

SIGNAL_REGISTRY_FULL.update(SIGNAL_REGISTRY)
SIGNAL_REGISTRY_FULL["hidden_norm"] = compute_hidden_norm
SIGNAL_REGISTRY_FULL["self_consistency"] = compute_self_consistency
SIGNAL_REGISTRY_FULL["verbalized"] = compute_verbalized_signal
SIGNAL_REGISTRY_FULL["neg_log_prob_correct"] = compute_neg_log_prob_correct


def register_signal(signal_type: str, fn: Callable) -> None:
    SIGNAL_REGISTRY_FULL[signal_type] = fn


def get_signal_fn(signal_type: str) -> Optional[Callable]:
    return SIGNAL_REGISTRY_FULL.get(signal_type)


def list_signals() -> list:
    return list(SIGNAL_REGISTRY_FULL.keys())


def compute_signal(signal_type: str, **kwargs) -> Any:
    fn = get_signal_fn(signal_type)
    if fn is None:
        raise ValueError(f"Unknown signal type: {signal_type}. Available: {list_signals()}")
    return fn(**kwargs)
