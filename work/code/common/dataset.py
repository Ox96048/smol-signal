import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from .types import Sample


def generate_checksum(data: str) -> str:
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def save_samples(samples: List[Sample], filepath: str) -> str:
    data = [asdict(sample) for sample in samples]
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    checksum = generate_checksum(json_str)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    return checksum


def load_samples(filepath: str) -> List[Sample]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Sample(**item) for item in data]


def validate_checksum(filepath: str, expected_checksum: str) -> bool:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    actual_checksum = generate_checksum(content)
    return actual_checksum == expected_checksum


def create_sample(prompt: str, answer: str, **kwargs) -> Sample:
    return Sample(
        prompt=prompt,
        answer=answer,
        meta=kwargs
    )