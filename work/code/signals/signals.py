"""MMQ 信号计算。所有函数输入 logits (Tensor [vocab_size])，输出标量 float。"""
import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> float:
    """
    logits: [vocab_size] float tensor
    returns: Shannon entropy in nats

    注意：用 log_softmax 计算数值稳定，避免 log(0)
    """
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum().item()
    return float(entropy)


def compute_margin(logits: torch.Tensor) -> float:
    """top1 和 top2 的 logit 差"""
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


def compute_max_prob(logits: torch.Tensor) -> float:
    """top1 的 softmax 概率（范围 0-1）"""
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    return float(F.softmax(logits, dim=-1).max().item())


def compute_neg_log_prob_correct(logits: torch.Tensor, correct_token_id: int) -> float:
    """
    正确答案 token 的负对数概率。用作"oracle"信号（需要知道答案）。
    不进 MMQ 主表，只用来诊断"信号理论上限"。
    """
    assert logits.dim() == 1, f"expected 1D logits, got shape {logits.shape}"
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-log_probs[correct_token_id].item())


SIGNAL_REGISTRY = {
    "entropy": compute_entropy,
    "margin": compute_margin,
    "max_prob": compute_max_prob,
}
