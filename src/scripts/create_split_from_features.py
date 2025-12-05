"""
Create split.json files for each precomputed features directory.

This script:
1. Scans each precomputed features directory (dinov2, dinov3) for subdirectories (take UIDs)
2. Randomly splits the takes into train/val/test sets
3. Saves separate split.json files for each features directory

Usage:
    python create_split_from_features.py --root ../../data/root --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --seed 42
"""

import argparse
import json
from pathlib import Path
import random
import numpy as np


def get_take_uids_from_features_dir(features_dir):
    """
    Extract all take UIDs (subdirectory names) from a features directory.
    
    Args:
        features_dir: Path to the precomputed features directory
        
    Returns:
        list: Sorted list of take UIDs (subdirectory names)
    """
    features_path = Path(features_dir)
    
    if not features_path.exists():
        print(f"WARNING: Features directory does not exist: {features_dir}")
        return []
    
    # Get all subdirectories (these are the take UIDs)
    take_uids = []
    for item in features_path.iterdir():
        if item.is_dir():
            take_uids.append(item.name)
    
    # Sort for reproducibility
    take_uids = sorted(take_uids)
    return take_uids


def create_random_split(take_uids, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Randomly split take UIDs into train/val/test sets.
    
    Args:
        take_uids: List of take UIDs to split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        dict: {'train': list, 'val': list, 'test': list} of take UIDs
    """
    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle take UIDs
    shuffled_uids = take_uids.copy()
    random.shuffle(shuffled_uids)
    
    total = len(shuffled_uids)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count  # Remaining goes to test
    
    # Split
    train_uids = sorted(shuffled_uids[:train_count])
    val_uids = sorted(shuffled_uids[train_count:train_count + val_count])
    test_uids = sorted(shuffled_uids[train_count + val_count:])
    
    return {
        'train': train_uids,
        'val': val_uids,
        'test': test_uids
    }


def save_split_json(split_dict, output_path, description=""):
    """
    Save split dictionary to JSON file.
    
    Args:
        split_dict: Dictionary with 'train', 'val', 'test' keys
        output_path: Path where to save the JSON file
        description: Optional description for logging
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    
    print(f"Saved {description}split.json to: {output_path}")
    print(f"  Train: {len(split_dict['train'])} takes")
    print(f"  Val: {len(split_dict['val'])} takes")
    print(f"  Test: {len(split_dict['test'])} takes")


def create_splits_from_features(args):
    """Main function to create split.json files from features directories."""
    
    root = Path(args.root)
    
    # Define features directories
    features_dirs = {
        'dinov2': root / 'precomputed_features_dinov2',
        'dinov3': root / 'precomputed_features_dinov3'
    }
    
    # Output paths for split.json files
    output_paths = {
        'dinov2': root / 'precomputed_features_dinov2' / 'split.json',
        'dinov3': root / 'precomputed_features_dinov3' / 'split.json'
    }
    
    print("=" * 60)
    print("CREATING SPLIT.JSON FILES FROM FEATURES DIRECTORIES")
    print("=" * 60)
    print(f"Root directory: {root}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Process each features directory
    for name, features_dir in features_dirs.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {name.upper()} features directory")
        print(f"{'=' * 60}")
        print(f"Features directory: {features_dir}")
        
        # Get all take UIDs from this features directory
        take_uids = get_take_uids_from_features_dir(features_dir)
        
        if len(take_uids) == 0:
            print(f"WARNING: No take UIDs found in {features_dir}")
            print(f"Skipping {name}...")
            continue
        
        print(f"Found {len(take_uids)} take UIDs")
        
        # Create random split
        split_dict = create_random_split(
            take_uids,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            seed=args.seed
        )
        
        # Save split.json
        output_path = output_paths[name]
        save_split_json(split_dict, output_path, description=f"{name.upper()} ")
        
        # Show some example UIDs
        print(f"\nExample take UIDs:")
        print(f"  Train (first 3): {split_dict['train'][:3]}")
        print(f"  Val (first 3): {split_dict['val'][:3]}")
        print(f"  Test (first 3): {split_dict['test'][:3]}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("\nSplit.json files created:")
    for name, output_path in output_paths.items():
        if output_path.exists():
            print(f"  ✓ {name}: {output_path}")
        else:
            print(f"  ✗ {name}: {output_path} (not created)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create split.json files from precomputed features directories"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of takes for training set (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of takes for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of takes for test set (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"ERROR: Ratios must sum to 1.0, got {total_ratio}")
        print(f"  Train: {args.train_ratio}")
        print(f"  Val: {args.val_ratio}")
        print(f"  Test: {args.test_ratio}")
        exit(1)
    
    create_splits_from_features(args)

