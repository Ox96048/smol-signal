"""verbalized 信号：语言化置信度（4 个模板）"""
from typing import List
import re


PROMPT_TEMPLATES = {
    "v1": {
        "system": "你是一个逻辑清晰的AI助手。请分析你对以下回答的信心程度。",
        "user": "问题: {question}\n回答: {answer}\n\n请用0-100的数字表示你对这个回答正确性的信心程度，只输出数字。"
    },
    "v2": {
        "system": "你是一个专家评估员。评估以下AI回答的可靠性。",
        "user": "任务: {question}\nAI的回答: {answer}\n\n这个回答正确的概率是多少？请用0到100之间的整数表示。"
    },
    "v3": {
        "system": "你正在评估AI生成内容的质量。",
        "user": "问题: {question}\n答案: {answer}\n\n在仔细分析后，你认为这个答案正确的置信度是多少百分比？只需要输出一个数字。"
    },
    "v4": {
        "system": "作为一个客观的评估者，评估以下回答的准确性。",
        "user": "Q: {question}\nA: {answer}\n\n请评估这个回答的正确性，用0-100的分数表示，0表示完全错误，100表示完全正确。"
    }
}


def build_prompt(template_name: str, question: str, answer: str) -> str:
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    template = PROMPT_TEMPLATES[template_name]
    return f"{template['system']}\n\n{template['user'].format(question=question, answer=answer)}"


def parse_confidence(response: str) -> float:
    match = re.search(r'(\d+)', response)
    if match:
        return min(max(int(match.group(1)) / 100.0, 0.0), 1.0)
    return 0.5


def compute_verbalized_signal(
    question: str,
    answer: str,
    response: str,
    template_version: str = "v1",
    **kwargs
) -> float:
    return parse_confidence(response)


def get_all_template_names() -> List[str]:
    return list(PROMPT_TEMPLATES.keys())
