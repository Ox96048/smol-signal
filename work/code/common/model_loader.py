from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel


def load_smollm(model_name: str = "HuggingFaceTB/SmolLM-135M") -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """加载 SmolLM base 模型 (非 instruct).

    Phase 0-3 用裸预训练模型, 不经过 instruction tuning.
    原因: instruct 模型的 verbalized confidence 有天花板效应，
    裸模型的 hidden states 更干净, 适合 probe 分析.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def load_model(model_name: str = "HuggingFaceTB/SmolLM-135M") -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    return load_smollm(model_name)


def get_model_output(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    return_hidden_states: bool = True,
    return_attentions: bool = False
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=return_hidden_states,
            output_attentions=return_attentions
        )
    return {
        "logits": outputs.logits,
        "hidden_states": outputs.hidden_states if return_hidden_states else None,
        "attentions": outputs.attentions if return_attentions else None,
        "last_hidden_state": outputs.hidden_states[-1] if return_hidden_states else None,
        "input_ids": inputs.input_ids
    }
