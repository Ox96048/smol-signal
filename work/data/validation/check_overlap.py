import json
import os
from typing import List, Dict, Set


def load_samples(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_prompts(samples: List[Dict]) -> Set[str]:
    return {sample['prompt'] for sample in samples}


def check_overlap(file1: str, file2: str) -> Dict:
    samples1 = load_samples(file1)
    samples2 = load_samples(file2)
    
    prompts1 = get_prompts(samples1)
    prompts2 = get_prompts(samples2)
    
    overlap = prompts1.intersection(prompts2)
    
    return {
        'file1': file1,
        'file2': file2,
        'count1': len(samples1),
        'count2': len(samples2),
        'overlap_count': len(overlap),
        'overlap_prompts': list(overlap)[:10]
    }


def check_all_overlaps(data_dir: str = '.') -> Dict:
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    results = []
    
    for i, file1 in enumerate(json_files):
        for j, file2 in enumerate(json_files):
            if i < j:
                result = check_overlap(os.path.join(data_dir, file1), os.path.join(data_dir, file2))
                results.append(result)
    
    return results


def generate_report(results: List[Dict], output_path: str = None) -> str:
    report = "# 数据集重叠检查报告\n\n"
    report += f"检查时间: {__import__('datetime').datetime.now().isoformat()}\n\n"
    report += "## 检查结果\n\n"
    
    for result in results:
        report += f"### {result['file1']} vs {result['file2']}\n"
        report += f"- 文件1样本数: {result['count1']}\n"
        report += f"- 文件2样本数: {result['count2']}\n"
        report += f"- 重叠数量: {result['overlap_count']}\n"
        if result['overlap_count'] > 0:
            report += "- 重叠示例:\n"
            for prompt in result['overlap_prompts']:
                report += f"  - {prompt[:50]}...\n"
        report += "\n"
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = check_all_overlaps(data_dir)
    report = generate_report(results, output_path)
    
    print(report)