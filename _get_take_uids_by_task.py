"""
Helper script to extract take UIDs from ego4d.json metadata for specific task IDs.
Usage: python get_take_ugids_by_task.py <metadata_dir> <task_id1> [task_id2] ...
"""

import json
import sys
from pathlib import Path


def get_take_uids_by_task(metadata_dir, task_ids):
    """
    Extract take UIDs that match the given task IDs.
    
    Args:
        metadata_dir: Path to directory containing ego4d.json and takes.json
        task_ids: List of task ID strings (e.g., ["1001", "1002"])
    
    Returns:
        List of take UIDs matching the task IDs
    """
    metadata_dir = Path(metadata_dir)
    
    # Load takes.json which contains task associations
    takes_file = metadata_dir / "takes.json"
    if not takes_file.exists():
        # Try v2 subdirectory
        takes_file = metadata_dir / "v2" / "takes.json"
    
    if not takes_file.exists():
        print(f"Error: Could not find takes.json in {metadata_dir}", file=sys.stderr)
        return []
    
    with open(takes_file, 'r') as f:
        takes_data = json.load(f)
    
    # Convert input task_ids to both string and int for comparison
    # (takes.json has task_id and parent_task_id as integers)
    task_ids_int = set()
    task_ids_str = set()
    for tid in task_ids:
        try:
            task_ids_int.add(int(tid))
        except ValueError:
            pass
        task_ids_str.add(str(tid))
    
    matching_uids = []
    
    # Iterate through takes and find those matching the task IDs
    if isinstance(takes_data, dict) and "takes" in takes_data:
        takes_list = takes_data["takes"]
    elif isinstance(takes_data, list):
        takes_list = takes_data
    else:
        print(f"Error: Unexpected format in takes.json", file=sys.stderr)
        return []
    
    for take in takes_list:
        # Check both task_id and parent_task_id fields
        # task_id and parent_task_id are integers in takes.json
        
        # Check if task_id matches (either the specific task or parent category)
        task_id = take.get("task_id")
        parent_task_id = take.get("parent_task_id")
        
        # Match if task_id OR parent_task_id is in our list
        matches = False
        if task_id is not None and (task_id in task_ids_int or str(task_id) in task_ids_str):
            matches = True
        elif parent_task_id is not None and (parent_task_id in task_ids_int or str(parent_task_id) in task_ids_str):
            matches = True
        
        if matches:
            # Extract take_uid (primary field name in takes.json)
            if "take_uid" in take:
                matching_uids.append(take["take_uid"])
            elif "uid" in take:
                matching_uids.append(take["uid"])
    
    return matching_uids


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python get_take_uids_by_task.py <metadata_dir> <task_id1> [task_id2] ...")
        print("\nExample:")
        print("  python get_take_uids_by_task.py outdir/metadata 1001 1002")
        sys.exit(1)
    
    metadata_dir = sys.argv[1]
    task_ids = sys.argv[2:]
    
    uids = get_take_uids_by_task(metadata_dir, task_ids)
    
    if uids:
        print(" ".join(uids))
    else:
        print(f"No takes found for task IDs: {', '.join(task_ids)}", file=sys.stderr)
        sys.exit(1)

