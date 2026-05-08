import os
import shutil
from datetime import datetime


def dump_snapshot(round_num: int, archive_dir: str = "archive/rounds") -> str:
    os.makedirs(archive_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(archive_dir, f"r{round_num:04d}_{timestamp}")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    files_to_copy = [
        "_STATE.md",
        "_TASKS.md",
        "_GUARDRAILS.md",
        "requirements.txt",
        "pytest.ini",
        "AGENT.md",
        "CLAUDE.md",
        "CODEX.md"
    ]
    
    for filename in files_to_copy:
        if os.path.exists(filename):
            shutil.copy(filename, os.path.join(snapshot_dir, filename))
    
    work_code_dir = "work/code"
    if os.path.exists(work_code_dir):
        dest_code_dir = os.path.join(snapshot_dir, "work", "code")
        shutil.copytree(work_code_dir, dest_code_dir, dirs_exist_ok=True)
    
    with open(os.path.join(snapshot_dir, "SNAPSHOT_INFO.md"), 'w', encoding='utf-8') as f:
        f.write(f"# Snapshot Info\n\n")
        f.write(f"## Round\n{r{round_num}}\n\n")
        f.write(f"## Timestamp\n{datetime.now().isoformat()}\n\n")
        f.write(f"## Files Included\n")
        for filename in files_to_copy:
            if os.path.exists(filename):
                f.write(f"- {filename}\n")
        f.write(f"- work/code/\n")
    
    return snapshot_dir


def clean_old_snapshots(archive_dir: str = "archive/rounds", keep_last: int = 10) -> None:
    if not os.path.exists(archive_dir):
        return
    
    dirs = sorted([d for d in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, d))])
    
    if len(dirs) <= keep_last:
        return
    
    dirs_to_delete = dirs[:-keep_last]
    for d in dirs_to_delete:
        shutil.rmtree(os.path.join(archive_dir, d))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python dump_snapshot.py <round_num>")
        sys.exit(1)
    
    try:
        round_num = int(sys.argv[1])
        snapshot_dir = dump_snapshot(round_num)
        print(f"Snapshot saved to: {snapshot_dir}")
        clean_old_snapshots()
    except ValueError:
        print("Error: round_num must be an integer")
        sys.exit(1)