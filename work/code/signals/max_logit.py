import torch
import torch.nn.functional as F
from typing import Optional
from common.types import SignalOutput


def compute_max_logit(logits: torch.Tensor, dim: int = -1) -> float:
    max_vals = torch.max(logits, dim=dim).values
    if max_vals.dim() > 0:
        max_vals = max_vals.mean()
    return max_vals.item()


def compute_max_prob(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    return float(probs.max().item())


def compute_max_logit_signal(
    logits: torch.Tensor,
    **kwargs
) -> SignalOutput:
    value = compute_max_logit(logits)
    return SignalOutput(
        value=value,
        confidence=1.0,
        metadata={
            "method": "max_logit",
            "logits_shape": logits.shape
        },
        raw_logits=logits
    )
