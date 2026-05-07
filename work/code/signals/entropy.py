"""MMQ 信号计算。所有函数输入 logits (Tensor [vocab_size])，输出标量 float。"""
import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> float:
    """
    logits: [vocab_size] float tensor
    returns: Shannon entropy in nats
    """
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum().item()
    return float(entropy)


def compute_entropy_signal(logits: torch.Tensor, **kwargs) -> float:
    return compute_entropy(logits)
