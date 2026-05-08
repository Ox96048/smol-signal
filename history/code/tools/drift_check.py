import difflib
import os
from typing import Dict, List, Tuple


def calculate_drift(anchor_content: str, current_content: str) -> Dict:
    anchor_lines = anchor_content.splitlines()
    current_lines = current_content.splitlines()
    
    matcher = difflib.SequenceMatcher(None, anchor_content, current_content)
    
    diff_ratio = matcher.ratio()
    drift_score = 1.0 - diff_ratio
    
    added_lines = []
    removed_lines = []
    modified_lines = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':
            added_lines.extend(current_lines[j1:j2])
        elif tag == 'delete':
            removed_lines.extend(anchor_lines[i1:i2])
        elif tag == 'replace':
            modified_lines.append({
                'original': ''.join(anchor_lines[i1:i2]),
                'modified': ''.join(current_lines[j1:j2])
            })
    
    return {
        'drift_score': min(drift_score, 1.0),
        'diff_ratio': diff_ratio,
        'added_lines': added_lines,
        'removed_lines': removed_lines,
        'modified_lines': modified_lines,
        'total_changes': len(added_lines) + len(removed_lines) + len(modified_lines)
    }


def run_audit(anchor_path: str, current_path: str, output_path: str = None) -> Dict:
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"Current file not found: {current_path}")
    
    with open(anchor_path, 'r', encoding='utf-8') as f:
        anchor_content = f.read()
    
    with open(current_path, 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    result = calculate_drift(anchor_content, current_content)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Drift Audit Report\n\n")
            f.write(f"## Audit Date\n{__import__('datetime').datetime.now().isoformat()}\n\n")
            f.write(f"## Drift Score\n{result['drift_score']:.4f}\n\n")
            f.write(f"## Diff Ratio\n{result['diff_ratio']:.4f}\n\n")
            f.write(f"## Total Changes\n{result['total_changes']}\n\n")
            
            if result['added_lines']:
                f.write("## Added Lines\n")
                for line in result['added_lines'][:20]:
                    f.write(f"- {line[:100]}...\n" if len(line) > 100 else f"- {line}\n")
                if len(result['added_lines']) > 20:
                    f.write(f"- ... and {len(result['added_lines']) - 20} more lines\n")
                f.write("\n")
            
            if result['removed_lines']:
                f.write("## Removed Lines\n")
                for line in result['removed_lines'][:20]:
                    f.write(f"- {line[:100]}...\n" if len(line) > 100 else f"- {line}\n")
                if len(result['removed_lines']) > 20:
                    f.write(f"- ... and {len(result['removed_lines']) - 20} more lines\n")
                f.write("\n")
            
            if result['modified_lines']:
                f.write("## Modified Sections\n")
                for i, mod in enumerate(result['modified_lines'][:10]):
                    f.write(f"### Change {i+1}\n")
                    f.write(f"**Original:** {mod['original'][:100]}...\n")
                    f.write(f"**Modified:** {mod['modified'][:100]}...\n\n")
                if len(result['modified_lines']) > 10:
                    f.write(f"... and {len(result['modified_lines']) - 10} more changes\n")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python drift_check.py <anchor_file> <current_file> [output_file]")
        sys.exit(1)
    
    anchor_path = sys.argv[1]
    current_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        result = run_audit(anchor_path, current_path, output_path)
        print(f"Drift Score: {result['drift_score']:.4f}")
        print(f"Total Changes: {result['total_changes']}")
        print(f"Added Lines: {len(result['added_lines'])}")
        print(f"Removed Lines: {len(result['removed_lines'])}")
        print(f"Modified Sections: {len(result['modified_lines'])}")
        
        if output_path:
            print(f"Report saved to: {output_path}")
        
        sys.exit(0 if result['drift_score'] <= 0.3 else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)