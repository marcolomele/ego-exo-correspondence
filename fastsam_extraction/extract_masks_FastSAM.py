"""
FastSAM Mask Extraction Script for EXO2EGO correspondence
Extracts object masks using FastSAM's Everything Mode from frames referenced in dataset JSON files.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import torch

# Add FastSAM to path (import deferred to avoid loading in dry-run mode)
fastsam_path = os.path.join(os.path.dirname(__file__), 'FastSAM')
sys.path.insert(0, fastsam_path)

# Configuration
REPO_ROOT = Path(__file__).parent.parent
O_MAMA_ROOT = REPO_ROOT / "o_mama_health"
PROCESSED_DIR = O_MAMA_ROOT / "processed"
DATASET_JSONS_DIR = O_MAMA_ROOT / "dataset_jsons"
FASTSAM_WEIGHTS = Path(__file__).parent / "FastSAM" / "FastSAM-x.pt"

# FastSAM hyperparameters
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu' # Apple Silicon M4 Pro
IMGSZ = 1024
CONF = 0.4
IOU = 0.9
RETINA_MASKS = True

# Split configurations for EXO2EGO direction
SPLITS = {
    'TRAIN': 'train_exoego_pairs.json',
    'VAL': 'val_exoego_pairs.json',
    'TEST': 'test_exoego_pairs.json'
}


def parse_path(path_str):
    """
    Parse a path string to extract take_id, camera, and frame_idx.
    Format: root//downscaled_takes//take_id//camera//object//rgb//frame_idx
    or: root//take_id//camera//object//rgb//frame_idx
    """
    parts = path_str.split('//')
    
    # Handle both formats (with or without 'downscaled_takes')
    if 'downscaled_takes' in parts:
        take_id_idx = parts.index('downscaled_takes') + 1
    else:
        # Find take_id (UUID format)
        take_id_idx = None
        for i, part in enumerate(parts):
            if len(part) == 36 and part.count('-') == 4:  # UUID format
                take_id_idx = i
                break
        if take_id_idx is None:
            raise ValueError(f"Could not find take_id in path: {path_str}")
    
    take_id = parts[take_id_idx]
    camera = parts[take_id_idx + 1]
    frame_idx = parts[-1]  # Last element is the frame index
    
    return take_id, camera, frame_idx


def load_pairs_from_json(json_path):
    """
    Load pairs from JSON file and extract unique EGO frames (for EXO2EGO direction).
    For EXO2EGO: pair format is [ego_path, ego_mask, exo_path, exo_mask]
    We process EGO frames (element at index 0).
    """
    with open(json_path, 'r') as f:
        pairs = json.load(f)
    
    unique_frames = set()
    for pair in pairs:
        # For EXO2EGO, the destination (EGO) is at index 0
        ego_path = pair[0]
        take_id, camera, frame_idx = parse_path(ego_path)
        unique_frames.add((take_id, camera, frame_idx))
    
    return sorted(list(unique_frames))


def extract_masks_and_boxes_from_results(results):
    """
    Extract masks and bounding boxes from FastSAM results.
    Returns:
        masks: numpy array of shape (N, H, W), dtype uint8
        boxes: numpy array of shape (N, 4), dtype float32, format [x1, y1, w, h]
    """
    masks_list = []
    boxes_list = []
    
    # FastSAM results structure
    if len(results) == 0 or results[0].masks is None:
        return np.zeros((0, 0, 0), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)
    
    result = results[0]  # Get first result
    
    # Extract masks
    if hasattr(result.masks, 'data'):
        masks_tensor = result.masks.data  # Shape: (N, H, W)
        masks_np = masks_tensor.cpu().numpy()
        masks_list = [mask for mask in masks_np]
    
    # Extract boxes (in xyxy format, need to convert to xywh)
    if hasattr(result.boxes, 'xyxy'):
        boxes_tensor = result.boxes.xyxy  # Shape: (N, 4) in [x1, y1, x2, y2]
        boxes_np = boxes_tensor.cpu().numpy()
        
        for box in boxes_np:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            boxes_list.append([x1, y1, w, h])
    
    if len(masks_list) == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)
    
    masks_array = np.stack(masks_list, axis=0).astype(np.uint8)  # (N, H, W)
    boxes_array = np.array(boxes_list, dtype=np.float32)  # (N, 4)
    
    return masks_array, boxes_array


def process_frame(model, frame_path, output_dir, take_id, camera, frame_idx):
    """
    Process a single frame: load image, run FastSAM, save masks and boxes.
    """
    # Create output directory structure
    output_subdir = output_dir / take_id / camera
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    masks_output = output_subdir / f"{frame_idx}_masks.npz"
    boxes_output = output_subdir / f"{frame_idx}_boxes.npy"
    if masks_output.exists() and boxes_output.exists():
        return True, "already_processed"
    
    # Load image
    if not frame_path.exists():
        return False, f"image_not_found: {frame_path}"
    
    image = cv2.imread(str(frame_path))
    if image is None:
        return False, f"failed_to_load: {frame_path}"
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Run FastSAM
        results = model(
            image_rgb,
            device=DEVICE,
            retina_masks=RETINA_MASKS,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU
        )
        
        # Extract masks and boxes
        masks_array, boxes_array = extract_masks_and_boxes_from_results(results)
        
        # Save masks as compressed .npz
        np.savez_compressed(str(masks_output), masks_array)
        
        # Save boxes as .npy
        np.save(str(boxes_output), boxes_array)
        
        return True, f"success: {len(masks_array)} masks"
    
    except Exception as e:
        return False, f"error: {str(e)}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract FastSAM masks for EXO2EGO correspondence')
    parser.add_argument('--dry-run', action='store_true', help='Count frames without processing')
    args = parser.parse_args()
    
    print("="*80)
    print("FastSAM Mask Extraction for EXO2EGO Correspondence")
    print("="*80)
    
    # Check if FastSAM weights exist
    if not FASTSAM_WEIGHTS.exists():
        print(f"ERROR: FastSAM weights not found at {FASTSAM_WEIGHTS}")
        return
    
    # Dry run mode - just count frames
    if args.dry_run:
        print("\n[DRY RUN MODE] Counting frames to process...")
        for split_name, json_filename in SPLITS.items():
            json_path = DATASET_JSONS_DIR / json_filename
            if json_path.exists():
                unique_frames = load_pairs_from_json(json_path)
                print(f"  {split_name}: {len(unique_frames)} unique frames")
        print("\nRun without --dry-run to start processing.")
        return
    
    # Initialize FastSAM model
    print(f"\n[1/4] Loading FastSAM model from {FASTSAM_WEIGHTS}...")
    print(f"      Device: {DEVICE}, Image Size: {IMGSZ}, Conf: {CONF}, IoU: {IOU}")
    
    # Import FastSAM only when needed
    from FastSAM import FastSAM
    model = FastSAM(str(FASTSAM_WEIGHTS))
    print("      Model loaded successfully!")
    
    # Process each split
    total_stats = {'success': 0, 'already_processed': 0, 'failed': 0, 'total_frames': 0}
    
    for split_name, json_filename in SPLITS.items():
        print(f"\n[2/4] Processing {split_name} split...")
        
        # Load pairs and extract unique frames
        json_path = DATASET_JSONS_DIR / json_filename
        if not json_path.exists():
            print(f"      WARNING: {json_path} not found, skipping...")
            continue
        
        print(f"      Loading pairs from {json_filename}...")
        unique_frames = load_pairs_from_json(json_path)
        print(f"      Found {len(unique_frames)} unique EGO frames to process")
        total_stats['total_frames'] += len(unique_frames)
        
        # Create output directory
        output_dir = O_MAMA_ROOT / f"Masks_{split_name}_EXO2EGO"
        output_dir.mkdir(exist_ok=True)
        print(f"      Output directory: {output_dir}")
        
        # Process each frame
        print(f"\n[3/4] Extracting masks for {split_name}...")
        success_count = 0
        already_processed = 0
        failed_count = 0
        failed_examples = []
        
        for take_id, camera, frame_idx in tqdm(unique_frames, desc=f"      {split_name}"):
            # Construct frame path in processed directory
            frame_path = PROCESSED_DIR / take_id / camera / f"{frame_idx}.jpg"
            
            success, message = process_frame(
                model, frame_path, output_dir, take_id, camera, frame_idx
            )
            
            if success:
                if "already_processed" in message:
                    already_processed += 1
                else:
                    success_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # Keep first 10 errors
                    failed_examples.append(f"{take_id}/{camera}/{frame_idx}: {message}")
        
        # Update total stats
        total_stats['success'] += success_count
        total_stats['already_processed'] += already_processed
        total_stats['failed'] += failed_count
        
        print(f"\n      {split_name} Results:")
        print(f"        - Successfully processed: {success_count}")
        print(f"        - Already processed: {already_processed}")
        print(f"        - Failed: {failed_count}")
        
        if failed_examples:
            print(f"\n      First {len(failed_examples)} errors:")
            for err in failed_examples:
                print(f"        - {err}")
    
    # Print final summary
    print("\n[4/4] Extraction complete!")
    print("="*80)
    print("FINAL SUMMARY:")
    print(f"  Total frames to process: {total_stats['total_frames']}")
    print(f"  Successfully processed: {total_stats['success']}")
    print(f"  Already processed (skipped): {total_stats['already_processed']}")
    print(f"  Failed: {total_stats['failed']}")
    if total_stats['failed'] > 0:
        failure_rate = 100 * total_stats['failed'] / total_stats['total_frames']
        print(f"  Failure rate: {failure_rate:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()

