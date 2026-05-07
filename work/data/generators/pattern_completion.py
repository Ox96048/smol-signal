import random
import json
import hashlib
from typing import List, Dict


def generate_pattern_completion(num_samples: int = 200) -> List[Dict]:
    patterns = [
        lambda n: [i for i in range(1, n+1)],
        lambda n: [i*2 for i in range(1, n+1)],
        lambda n: [i*3 for i in range(1, n+1)],
        lambda n: [2**i for i in range(n)],
        lambda n: [i**2 for i in range(1, n+1)],
        lambda n: list(reversed(range(1, n+1))),
        lambda n: [i if i % 2 == 1 else i*2 for i in range(1, n+1)],
        lambda n: [i*2-1 for i in range(1, n+1)]
    ]
    
    samples = []
    
    for i in range(num_samples):
        pattern_func = random.choice(patterns)
        length = random.randint(4, 8)
        sequence = pattern_func(length)
        hide_index = random.randint(1, length-2)
        answer = sequence[hide_index]
        sequence_with_hide = sequence[:hide_index] + ['?'] + sequence[hide_index+1:]
        
        prompt = " ".join(str(x) for x in sequence_with_hide) + " 下一个数字是什么？"
        
        samples.append({
            "id": f"pc_{i:04d}",
            "prompt": prompt,
            "answer": str(answer),
            "meta": {
                "type": "pattern_completion",
                "pattern_type": patterns.index(pattern_func),
                "sequence_length": length,
                "hidden_position": hide_index
            }
        })
    
    return samples


def generate_and_save(filepath: str, num_samples: int = 200) -> str:
    samples = generate_pattern_completion(num_samples)
    
    json_str = json.dumps(samples, ensure_ascii=False, indent=2)
    checksum = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return checksum


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pattern_completion.json"
    
    checksum = generate_and_save(output_path, num_samples)
    print(f"Generated {num_samples} samples")
    print(f"Output: {output_path}")
    print(f"Checksum: {checksum}")