from dataclasses import dataclass
from typing import Dict, Optional, Union, List
import torch


@dataclass
class SignalOutput:
    value: float
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None
    raw_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None


@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None


@dataclass
class Sample:
    prompt: str
    answer: str
    meta: Optional[Dict] = None
    signal_outputs: Optional[Dict[str, SignalOutput]] = None


@dataclass
class MMQResult:
    sample_id: str
    signal_values: Dict[str, float]
    mmq_score: float
    baseline_entropy: float
    auroc: Optional[float] = None
    confidence_interval: Optional[List[float]] = None