import random
import json
import hashlib
from typing import List, Dict


def generate_simple_logic(num_samples: int = 200) -> List[Dict]:
    logic_types = [
        ("如果今天下雨，地面会变___。", "湿"),
        ("鸟有翅膀，所以鸟会___。", "飞"),
        ("鱼生活在___里。", "水"),
        ("太阳从___方升起。", "东"),
        ("冰在___下会融化。", "0度"),
        ("人需要___来呼吸。", "空气"),
        ("三角形有___条边。", "三"),
        ("一年有___个季节。", "四"),
        ("正方形有___个角。", "四"),
        ("月亮在___亮。", "晚上")
    ]
    
    samples = []
    
    for i in range(num_samples):
        question, answer = random.choice(logic_types)
        prompt = question
        
        samples.append({
            "id": f"sl_{i:04d}",
            "prompt": prompt,
            "answer": answer,
            "meta": {
                "type": "simple_logic",
                "logic_index": logic_types.index((question, answer))
            }
        })
    
    return samples


def generate_and_save(filepath: str, num_samples: int = 200) -> str:
    samples = generate_simple_logic(num_samples)
    
    json_str = json.dumps(samples, ensure_ascii=False, indent=2)
    checksum = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return checksum


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    output_path = sys.argv[2] if len(sys.argv) > 2 else "simple_logic.json"
    
    checksum = generate_and_save(output_path, num_samples)
    print(f"Generated {num_samples} samples")
    print(f"Output: {output_path}")
    print(f"Checksum: {checksum}")