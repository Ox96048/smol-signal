import random
import json
import hashlib
from typing import List, Dict


def generate_d2_constructor(num_samples: int = 500) -> List[Dict]:
    samples = []
    
    word_pairs = [
        ("苹果", "水果", "蔬菜"),
        ("狗", "动物", "植物"),
        ("桌子", "家具", "食物"),
        ("书", "文具", "玩具"),
        ("汽车", "交通工具", "建筑物"),
        ("花", "植物", "动物"),
        ("电脑", "电子产品", "食品"),
        ("鞋子", "服饰", "电器"),
        ("杯子", "餐具", "工具"),
        ("树", "植物", "动物")
    ]
    
    for i in range(num_samples):
        word, correct, wrong = random.choice(word_pairs)
        if random.random() > 0.5:
            prompt = f"{word}是一种{wrong}吗？"
            answer = "不是"
        else:
            prompt = f"{word}是一种{correct}吗？"
            answer = "是"
        
        samples.append({
            "id": f"d2_{i:04d}",
            "prompt": prompt,
            "answer": answer,
            "meta": {
                "type": "d2_constructor",
                "word": word,
                "correct_category": correct,
                "wrong_category": wrong,
                "is_correct": answer == "是"
            }
        })
    
    return samples


def generate_and_save(filepath: str, num_samples: int = 500) -> str:
    samples = generate_d2_constructor(num_samples)
    
    json_str = json.dumps(samples, ensure_ascii=False, indent=2)
    checksum = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return checksum


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_path = sys.argv[2] if len(sys.argv) > 2 else "d2_constructor.json"
    
    checksum = generate_and_save(output_path, num_samples)
    print(f"Generated {num_samples} samples")
    print(f"Output: {output_path}")
    print(f"Checksum: {checksum}")