"""答案 token 定位与提取工具。

关键洞察 (建议与批评 §1-2):
  - "12+7? The answer is" 的第一个生成 token 可能是空格/标点,
    不是答案 token. 直接取第 0 个 token 的 entropy 测的是无意义噪声.
  - SmolLM 的 BPE tokenizer 会把 "42" 拆成 "4"+"2",
    需要知道 answer 跨几个 token 才能正确构造 span_loss_target.
"""

import re
import torch
from typing import List, Optional, Tuple


def find_answer_start_position(
    input_ids: torch.Tensor,
    tokenizer,
    prompt: str,
    answer_separator: str = "The answer is",
) -> int:
    """找到 prompt 中 answer 应该在的位置 (最后一个 prompt token 之后).

    返回: answer 起始位置在 input_ids 中的索引.
    如果 prompt 占据整个 input_ids, 返回 len(input_ids) (即需要生成).
    """
    sep_ids = tokenizer.encode(answer_separator, add_special_tokens=False)
    seq = input_ids.squeeze().tolist() if input_ids.dim() > 1 else input_ids.tolist()

    for i in range(len(seq) - len(sep_ids) + 1):
        if seq[i:i + len(sep_ids)] == sep_ids:
            return i + len(sep_ids)

    return len(seq)


def get_answer_token_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    answer_token_offset: int = 0,
) -> torch.Tensor:
    """提取 answer span 的 logits.

    参数:
      logits: [seq_len, vocab_size] 或 [1, seq_len, vocab_size]
      input_ids: prompt token ids
      generated_ids: 模型生成的 token ids (接在 input_ids 后面)
      answer_token_offset: 跳过前几个生成 token (如空格/换行).

    返回:
      answer_logits: [n_answer_tokens, vocab_size]
      answer_start_pos: answer 在 logits 中的起始位置

    注意: logits[i] 是预测 token i+1 的分数,
      所以 answer 的 logits 在 input_len + offset 位置开始.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)

    input_len = input_ids.shape[-1] if input_ids.dim() > 1 else len(input_ids)
    gen_len = generated_ids.shape[-1] if generated_ids.dim() > 1 else len(generated_ids)

    answer_start = input_len + answer_token_offset
    answer_end = input_len + gen_len

    if answer_start >= logits.shape[0]:
        raise ValueError(
            f"answer_start={answer_start} >= logits.shape[0]={logits.shape[0]}. "
            f"input_len={input_len}, offset={answer_token_offset}"
        )

    answer_end = min(answer_end, logits.shape[0])
    return logits[answer_start:answer_end], answer_start


def strip_non_answer_tokens(
    generated_text: str,
    tokenizer,
    answer_patterns: Optional[List[str]] = None,
) -> str:
    """从生成文本中剥离非答案 token (空格、换行、格式字符等).

    标准方法: 用正则提取预期格式的答案.
    """
    if answer_patterns is None:
        answer_patterns = [
            r"(-?\d+)",                     # 数字
            r"\b(yes|no)\b",               # yes/no
            r"\b(true|false)\b",           # true/false
            r"\b([A-E])\b",                # 选项
            r"\b(larger|smaller|equal)\b", # 比较
        ]

    text = generated_text.strip()
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return text


def get_answer_token_count(
    answer: str,
    tokenizer,
) -> int:
    """返回 answer 被 tokenizer 拆成几个 token."""
    ids = tokenizer.encode(answer, add_special_tokens=False)
    return len(ids)


def compute_answer_span_loss(
    logits: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_start_pos: int,
    reduction: str = "sum",
) -> float:
    """计算 answer span 的累积 loss.

    参数:
      logits: [seq_len, vocab_size] 模型输出的完整 logits
      answer_ids: answer 的 token ids [n_tokens]
      answer_start_pos: answer 在 logits 中的起始位置
      reduction: "sum" | "mean" | "last" | "log_mean"

    返回: float
    """
    n_tokens = len(answer_ids)
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    losses = []
    for i in range(n_tokens):
        pos = answer_start_pos + i
        if pos >= logits.shape[0]:
            break
        loss = ce_loss(logits[pos:pos + 1], answer_ids[i:i + 1])
        losses.append(loss.item())

    if not losses:
        return 0.0

    if reduction == "sum":
        return sum(losses)
    elif reduction == "mean":
        return sum(losses) / len(losses)
    elif reduction == "last":
        return losses[-1]
    elif reduction == "log_mean":
        import math
        mean_loss = sum(losses) / len(losses)
        return math.log(mean_loss + 1.0)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
