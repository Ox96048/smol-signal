"""margin 信号：top1 和 top2 的 logit 差"""
import torch


def compute_margin(logits: torch.Tensor) -> float:
    """top1 和 top2 的 logit 差"""
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


def compute_margin_signal(logits: torch.Tensor, **kwargs) -> float:
    return compute_margin(logits)
