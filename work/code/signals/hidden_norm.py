"""hidden_norm 信号：最后一层最后一个 token 的范数

注意：输入是 hidden_states 列表，取 last layer + last token。
"""
import torch
from typing import List


def compute_hidden_norm(hidden_states: List[torch.Tensor], p: int = 2) -> float:
    """
    hidden_states: 模型输出的隐藏层列表，hidden_states[-1] 是最后一层
    取最后一层的最后一个 token，计算 Lp 范数。
    """
    last_layer = hidden_states[-1]
    assert last_layer.dim() == 3, f"expected [batch, seq, dim], got {last_layer.shape}"
    last_token = last_layer[0, -1, :]
    return float(torch.norm(last_token, p=p).item())


def compute_hidden_norm_signal(hidden_states: List[torch.Tensor], **kwargs) -> float:
    return compute_hidden_norm(hidden_states)
