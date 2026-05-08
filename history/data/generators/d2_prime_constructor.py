import random
import json
import hashlib
from typing import List, Dict


def generate_d2_prime_constructor(num_samples: int = 500) -> List[Dict]:
    samples = []
    
    templates = [
        ("{num1}加{num2}等于多少？", lambda x, y: str(x + y)),
        ("{num1}减{num2}等于多少？", lambda x, y: str(x - y)),
        ("{num1}乘以{num2}等于多少？", lambda x, y: str(x * y)),
        ("{num1}除以{num2}等于多少？", lambda x, y: str(x // y))
    ]
    
    for i in range(num_samples):
        template, func = random.choice(templates)
        num1 = random.randint(1, 20)
        
        if template == "{num1}除以{num2}等于多少？":
            divisors = [d for d in range(1, num1 + 1) if num1 % d == 0]
            num2 = random.choice(divisors) if divisors else 1
        else:
            num2 = random.randint(1, 20)
        
        answer = func(num1, num2)
        prompt = template.format(num1=num1, num2=num2)
        
        samples.append({
            "id": f"d2p_{i:04d}",
            "prompt": prompt,
            "answer": answer,
            "meta": {
                "type": "d2_prime_constructor",
                "num1": num1,
                "num2": num2,
                "operation": template.split("{")[0].strip()
            }
        })
    
    return samples


def generate_and_save(filepath: str, num_samples: int = 500) -> str:
    samples = generate_d2_prime_constructor(num_samples)
    
    json_str = json.dumps(samples, ensure_ascii=False, indent=2)
    checksum = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return checksum


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_path = sys.argv[2] if len(sys.argv) > 2 else "d2_prime_constructor.json"
    
    checksum = generate_and_save(output_path, num_samples)
    print(f"Generated {num_samples} samples")
    print(f"Output: {output_path}")
    print(f"Checksum: {checksum}")