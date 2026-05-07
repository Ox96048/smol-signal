import random
import json
import hashlib
from typing import List, Dict


def generate_arithmetic_easy(num_samples: int = 200) -> List[Dict]:
    samples = []
    
    for i in range(num_samples):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        answer = a + b
        
        prompt = f"{a} + {b} = ?"
        samples.append({
            "id": f"ae_{i:04d}",
            "prompt": prompt,
            "answer": str(answer),
            "meta": {
                "type": "arithmetic_easy",
                "a": a,
                "b": b,
                "operation": "addition"
            }
        })
    
    return samples


def generate_and_save(filepath: str, num_samples: int = 200) -> str:
    samples = generate_arithmetic_easy(num_samples)
    
    json_str = json.dumps(samples, ensure_ascii=False, indent=2)
    checksum = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return checksum


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    output_path = sys.argv[2] if len(sys.argv) > 2 else "arithmetic_easy.json"
    
    checksum = generate_and_save(output_path, num_samples)
    print(f"Generated {num_samples} samples")
    print(f"Output: {output_path}")
    print(f"Checksum: {checksum}")