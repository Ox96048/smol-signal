import re
from typing import Dict, Any, Optional


PHASES = [
    "bootstrap-infrastructure",
    "phase-minus-1",
    "phase-0",
    "phase-1",
    "phase-2",
    "phase-3",
    "paused-awaiting-human",
    "circuit-broken"
]


def validate_state(content: str) -> Dict[str, Any]:
    errors = []
    warnings = []
    
    if not content.strip():
        errors.append("STATE 文件为空")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    round_match = re.search(r'round:\s*(\d+)', content)
    if not round_match:
        errors.append("缺少 round 字段")
    else:
        try:
            round_num = int(round_match.group(1))
            if round_num < 1:
                errors.append(f"round 必须 >= 1，当前值: {round_num}")
        except ValueError:
            errors.append(f"round 必须是整数，当前值: {round_match.group(1)}")
    
    phase_match = re.search(r'phase:\s*(\S+)', content)
    if not phase_match:
        errors.append("缺少 phase 字段")
    else:
        phase = phase_match.group(1)
        if phase not in PHASES:
            errors.append(f"phase 值无效: {phase}，有效值: {', '.join(PHASES)}")
    
    drift_match = re.search(r'drift_score:\s*([\d.]+)', content)
    if drift_match:
        try:
            drift_score = float(drift_match.group(1))
            if drift_score < 0 or drift_score > 1:
                warnings.append(f"drift_score 超出合理范围 [0,1]: {drift_score}")
        except ValueError:
            errors.append(f"drift_score 必须是数字，当前值: {drift_match.group(1)}")
    
    status_match = re.search(r'last_round_status:\s*([✅⚠️❌])', content)
    if not status_match:
        errors.append("缺少 last_round_status 字段或格式不正确")
    
    consecutive_failures_match = re.search(r'consecutive_failures:\s*(\d+)', content)
    if consecutive_failures_match:
        try:
            failures = int(consecutive_failures_match.group(1))
            if failures < 0:
                errors.append(f"consecutive_failures 不能为负: {failures}")
        except ValueError:
            errors.append(f"consecutive_failures 必须是整数")
    
    next_audit_match = re.search(r'next_audit:\s*(\d+)', content)
    if next_audit_match:
        try:
            next_audit = int(next_audit_match.group(1))
            if next_audit < 1:
                errors.append(f"next_audit 必须 >= 1: {next_audit}")
        except ValueError:
            errors.append(f"next_audit 必须是整数")
    
    if not re.search(r'task_id:\s*T-\d+', content):
        errors.append("缺少有效的 task_id 字段")
    
    if not re.search(r'title:\s*', content):
        errors.append("缺少 title 字段")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def validate_tasks(content: str) -> Dict[str, Any]:
    errors = []
    warnings = []
    
    if not content.strip():
        errors.append("TASKS 文件为空")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    active_tasks = re.findall(r'\[active\]', content)
    if len(active_tasks) != 1:
        errors.append(f"必须有且仅有一个 active 任务，当前数量: {len(active_tasks)}")
    
    task_ids = re.findall(r'T-\d+', content)
    if len(task_ids) != len(set(task_ids)):
        errors.append("存在重复的 task_id")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_state.py <state_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "round:" in content:
            result = validate_state(content)
            print("STATE 文件校验结果:")
        else:
            result = validate_tasks(content)
            print("TASKS 文件校验结果:")
        
        print(f"有效: {result['valid']}")
        if result['errors']:
            print("错误:")
            for err in result['errors']:
                print(f"  - {err}")
        if result['warnings']:
            print("警告:")
            for warn in result['warnings']:
                print(f"  - {warn}")
        
        sys.exit(0 if result['valid'] else 1)
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {filepath}")
        sys.exit(1)