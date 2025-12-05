"""
Extract take UIDs from relation annotation files based on filtering criteria.

Usage:
    python filter_annotations.py
    
The script processes all JSON files in outdir/relation_annotations/ except splits.json,
and extracts UIDs that match the following criteria:
- Scenario is in cooking_scenarios or health_scenarios
- object_masks contains at least one element
- For at least one object in object_masks, there are both:
  - An exo perspective camera (key does NOT contain 'aria')
  - An ego perspective camera (key contains 'aria')

Output:
    - target_uids_cooking.txt: UIDs for cooking scenarios
    - target_uids_health.txt: UIDs for health scenarios
"""

import json
import sys
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm


# Scenario lists
health_scenarios = ["Covid-19 Rapid Antigen Test"]

cooking_scenarios = [
    "Cooking",
    "Cooking an Omelet",
    "Cooking Scrambled Eggs",
    "Cooking Tomato & Eggs",
    "Cooking Noodles",
    "Cooking Dumplings",
    "Cooking Noodles",
    "Cooking Pasta",
    "Cooking Sushi Rolls",
    "Cooking Samosas",
    "Making Cucumber & Tomato Salad",
    "Making Sesame-Ginger Asian Salad",
    "Making Greek Salad",
    "Making Coffee latte",
    "Making Chai Tea",
    "Making Milk Tea",
    "Cooking Cookies",
    "Cooking Brownies",
    "Making White Radish & Lettuce & Tomato & Cucumber Salad"
]


def check_object_masks_has_both_perspectives(object_masks: Dict[str, Any]) -> bool:
    """
    Check if at least one object in object_masks has both exo and ego perspectives.
    
    Exo perspective: camera key does NOT contain 'aria'
    Ego perspective: camera key contains 'aria'
    """
    if not object_masks:
        return False
    
    for object_name, cameras in object_masks.items():
        if not isinstance(cameras, dict):
            continue
        
        has_exo = False
        has_ego = False
        
        for camera_key in cameras.keys():
            if 'aria' in camera_key.lower():
                has_ego = True
            else:
                has_exo = True
        
        if has_exo and has_ego:
            return True
    
    return False


def get_split_from_filename(filename: str) -> str:
    """Extract split name (train/val/test) from filename."""
    if 'train' in filename.lower():
        return 'train'
    elif 'val' in filename.lower():
        return 'val'
    elif 'test' in filename.lower():
        return 'test'
    return 'unknown'


def process_annotation_file(file_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Process a single annotation file and return dictionaries of matching UIDs grouped by split.
    
    Returns:
        Tuple of (cooking_uids_by_split, health_uids_by_split)
        Each dict has keys: 'train', 'val', 'test'
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    split = get_split_from_filename(file_path.name)
    
    cooking_uids_by_split = defaultdict(list)
    health_uids_by_split = defaultdict(list)
    
    health_scenarios_set = set(health_scenarios)
    cooking_scenarios_set = set(cooking_scenarios)
    
    for uid, annotation in tqdm(annotations.items(), desc=f"Processing {file_path.name}", leave=False):
        # Check scenario
        scenario = annotation.get('scenario', '')
        
        # Determine scenario type
        is_health = scenario in health_scenarios_set
        is_cooking = scenario in cooking_scenarios_set
        
        if not (is_health or is_cooking):
            continue
        
        # Check object_masks has at least one element
        object_masks = annotation.get('object_masks', {})
        if not object_masks:
            continue
        
        # Check for both perspectives
        if check_object_masks_has_both_perspectives(object_masks):
            if is_health:
                health_uids_by_split[split].append(uid)
            if is_cooking:
                cooking_uids_by_split[split].append(uid)
    
    return cooking_uids_by_split, health_uids_by_split


def main():
    relation_annotations_dir = Path("outdir/relation_annotations")
    
    if not relation_annotations_dir.exists():
        print(f"Error: Directory not found: {relation_annotations_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Get all JSON files except splits.json
    json_files = [
        f for f in relation_annotations_dir.glob("*.json")
        if f.name != "splits.json"
    ]
    
    if not json_files:
        print(f"Error: No JSON files found in {relation_annotations_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(json_files)} annotation files to process\n")
    
    # Collect all matching UIDs grouped by split
    all_cooking_uids_by_split = defaultdict(list)
    all_health_uids_by_split = defaultdict(list)
    
    for json_file in tqdm(sorted(json_files), desc="Processing files"):
        cooking_uids_by_split, health_uids_by_split = process_annotation_file(json_file)
        for split, uids in cooking_uids_by_split.items():
            all_cooking_uids_by_split[split].extend(uids)
        for split, uids in health_uids_by_split.items():
            all_health_uids_by_split[split].extend(uids)
    
    # Remove duplicates while preserving order for each split
    def deduplicate_uids(uids_by_split: Dict[str, List[str]]) -> Dict[str, List[str]]:
        unique_uids_by_split = {}
        for split, uids in uids_by_split.items():
            unique_uids = []
            seen = set()
            for uid in uids:
                if uid not in seen:
                    unique_uids.append(uid)
                    seen.add(uid)
            unique_uids_by_split[split] = unique_uids
        return unique_uids_by_split
    
    unique_cooking_uids_by_split = deduplicate_uids(all_cooking_uids_by_split)
    unique_health_uids_by_split = deduplicate_uids(all_health_uids_by_split)
    
    # Save to files with separators
    cooking_output_file = Path("outdir/target_uids_cooking.txt")
    health_output_file = Path("outdir/target_uids_health.txt")
    
    def write_uids_by_split(output_file: Path, uids_by_split: Dict[str, List[str]]):
        """Write UIDs grouped by split with separator lines."""
        splits_order = ['train', 'val', 'test']
        total_count = 0
        
        with open(output_file, 'w') as f:
            for split in splits_order:
                if split in uids_by_split and uids_by_split[split]:
                    uids = uids_by_split[split]
                    f.write(f"# {split.upper()}\n")
                    for uid in uids:
                        f.write(f"{uid}\n")
                    f.write("\n")  # Empty line separator
                    total_count += len(uids)
        
        return total_count
    
    cooking_total = write_uids_by_split(cooking_output_file, unique_cooking_uids_by_split)
    health_total = write_uids_by_split(health_output_file, unique_health_uids_by_split)
    
    print(f"\nTotal unique cooking UIDs: {cooking_total}")
    for split in ['train', 'val', 'test']:
        if split in unique_cooking_uids_by_split:
            count = len(unique_cooking_uids_by_split[split])
            print(f"  {split}: {count}")
    print(f"Saved to: {cooking_output_file}")
    
    print(f"\nTotal unique health UIDs: {health_total}")
    for split in ['train', 'val', 'test']:
        if split in unique_health_uids_by_split:
            count = len(unique_health_uids_by_split[split])
            print(f"  {split}: {count}")
    print(f"Saved to: {health_output_file}")


if __name__ == "__main__":
    main()
