#!/usr/bin/env python3
"""
Download and process EgoExo4D video takes for correspondence task.

This script downloads videos, extracts annotated frames, and organizes them
for baseline model training.

Usage:
    python download_and_process_data.py --scenario cooking
    python download_and_process_data.py --scenario health
"""

import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import cv2
from decord import VideoReader, cpu
from lzstring import LZString


def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def decode_mask(width: int, height: int, encoded_mask: str):
    """Decode LZString compressed mask to COCO RLE format."""
    try:
        decomp_string = LZString.decompressFromEncodedURIComponent(encoded_mask)
    except:
        return None
    
    decomp_encoded = decomp_string.encode()
    rle_obj = {
        "size": [height, width],
        "counts": decomp_encoded,
    }
    rle_obj['counts'] = rle_obj['counts'].decode('ascii')
    return rle_obj


def process_masks(object_masks: Dict[str, Any]):
    """
    Process and decode all masks in object_masks annotation in format: {object_id}/{cam_id}/{frame_id} -> COCO RLE
    """
    processed_masks = {}
    
    for object_id, obj_data in object_masks.items():
        processed_masks[object_id] = {}
        
        for cam_id, cam_data in obj_data.items():
            processed_masks[object_id][cam_id] = {}
            
            annotations = cam_data.get("annotation", {})
            for frame_id, frame_data in annotations.items():
                width = frame_data.get("width")
                height = frame_data.get("height")
                encoded_mask = frame_data.get("encodedMask")
                
                if width and height and encoded_mask:
                    coco_mask = decode_mask(width, height, encoded_mask)
                    if coco_mask:
                        processed_masks[object_id][cam_id][frame_id] = coco_mask
    
    return processed_masks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and process EgoExo4D data for correspondence task"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["cooking", "health"],
        help="Scenario to process (cooking or health)"
    )
    parser.add_argument(
        "--annotation-root",
        type=str,
        default="../annotations/relation_annotations",
        help="Root directory for relation annotations"
    )
    return parser.parse_args()


def load_split_uids(scenario: str) -> Dict[str, List[str]]:
    """Load UIDs from split_{scenario}.json file."""
    split_file = Path(f"../output_dir_{scenario}/split.json")
    
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    print(f"Loaded splits for {scenario}:")
    for split_name, uids in splits.items():
        print(f"  {split_name}: {len(uids)} UIDs")
    
    return splits


def find_annotation_for_uid(uid: str, annotation_root: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """Search for annotation by UID across all relation annotation files."""
    annotation_files = [
        ("relations_train.json", "train"),
        ("relations_val.json", "val"),
        ("relations_test.json", "test")
    ]
    
    for filename, split_name in annotation_files:
        file_path = Path(annotation_root) / filename
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if uid in data.get('annotations', {}):
                return data['annotations'][uid], split_name
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return None


def download_take(uid: str, output_dir: str) -> bool:
    """Download take using egoexo CLI with visible progress bar."""
    cmd = [
        "egoexo",
        "-o", output_dir,
        "--parts", 
        #"downscaled_takes/448",
        "takes",
        "--yes",
        "--uids", uid
    ]
    
    print(f"\n{'─'*60}")
    print(f"Downloading take: {uid}")
    print(f"{'─'*60}")
    
    try:
        # Don't capture output so progress bar is visible
        subprocess.run(cmd, check=True)
        print(f"Download complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def get_all_cameras_from_annotation(annotation: Dict[str, Any]) -> Dict[str, List[int]]:
    """Extract all cameras and their annotated frames from annotation."""
    camera_frames = {}
    
    for obj_data in annotation.get('object_masks', {}).values():
        for cam_name, cam_data in obj_data.items():
            if cam_name not in camera_frames:
                camera_frames[cam_name] = set()
            camera_frames[cam_name].update(cam_data.get('annotated_frames', []))
    
    return {cam: sorted(list(frames)) for cam, frames in camera_frames.items()}


def extract_frames_from_video(
    video_path: Path,
    frame_indices: List[int],
    output_dir: Path,
    is_aria: bool = False
) -> bool:
    """Extract specific frames from video and save as JPEGs."""
    if not video_path.exists():
        print(f"Video not found: {video_path.name}")
        return False
    
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        valid_indices = [idx for idx in frame_indices if idx < len(vr)]
        
        if not valid_indices:
            print(f"No valid frame indices for {video_path.name}")
            return False
        
        if len(valid_indices) < len(frame_indices):
            print(f"Warning: {len(frame_indices) - len(valid_indices)} frame indices out of bounds for {video_path}")
        
        frames = vr.get_batch(valid_indices).asnumpy()
        scale = 2 if is_aria else 4
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for frame, idx in zip(frames, valid_indices):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            H, W = frame_bgr.shape[:2]
            frame_resized = cv2.resize(frame_bgr, (W // scale, H // scale))
            cv2.imwrite(str(output_dir / f"{idx}.jpg"), frame_resized)
        
        print(f"Extracted {len(valid_indices)} frames")
        return True
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return False


def process_take(
    uid: str,
    annotation: Dict[str, Any],
    output_dir: Path
) -> bool:
    """Process a single take: download, extract frames, create annotation.json."""
    take_start = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"Processing: {uid}")
    print(f"{'='*80}")
    
    take_name = annotation.get('take_name')
    if not take_name:
        print(f"Error: No take_name found in annotation for {uid}")
        return False
    
    print(f"Take name: {take_name}")
    
    # Download
    if not download_take(uid, str(output_dir)):
        return False
    
    # Find downloaded videos
    takes_dir = output_dir / "takes" / take_name / "frame_aligned_videos" # / "downscaled" / "448"
    
    if not takes_dir.exists():
        print(f"Error: Downloaded videos not found at {takes_dir}")
        # Try alternative path without frame_aligned_videos
        takes_dir = output_dir / "takes" / take_name #/ "downscaled" / "448"
        if not takes_dir.exists():
            print(f"Videos not found at expected locations")
            return False
    
    # Get cameras and frames
    camera_frames = get_all_cameras_from_annotation(annotation)
    if not camera_frames:
        print(f"Warning: No cameras found in annotation for {uid}")
        return False
    
    print(f"Found {len(camera_frames)} cameras to process")
    
    # Create take output directory
    take_output_dir = output_dir / uid
    take_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process cameras
    success_count = 0
    for cam_name, frame_indices in camera_frames.items():
        print(f"\nProcessing camera: {cam_name}")
        print(f"  Annotated frames: {len(frame_indices)}")
        
        # Find video file
        video_path = takes_dir / f"{cam_name}.mp4"
        is_aria = 'aria' in cam_name.lower()
        
        # Extract frames
        cam_output_dir = take_output_dir / cam_name
        if extract_frames_from_video(video_path, frame_indices, cam_output_dir, is_aria):
            success_count += 1
    
    if success_count == 0:
        print(f"Error: No frames extracted for {uid}")
        return False
    
    print(f"\nSuccessfully processed {success_count}/{len(camera_frames)} cameras")
    
    # Process masks - decode from LZString to COCO RLE format
    print("\nProcessing and decoding masks...")
    object_masks = annotation.get("object_masks", {})
    decoded_masks = process_masks(object_masks)
    print(f"Decoded masks for {len(decoded_masks)} objects")
    
    # Get all extracted frame indices (union of all cameras)
    all_frame_indices = set()
    for frames in camera_frames.values():
        all_frame_indices.update(frames)
    subsample_idx = sorted(list(all_frame_indices))
    
    # Create annotation.json
    annotation_output = {
        "scenario": annotation.get("scenario", ""),
        "take_name": take_name,
        "object_masks": object_masks,
        "masks": decoded_masks,
        "subsample_idx": subsample_idx
    }
    
    annotation_path = take_output_dir / "annotation.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation_output, f, indent=2)
    
    print(f"Created annotation file: {annotation_path}")
    
    # Cleanup: Delete takes directory
    takes_root = output_dir / "takes"
    if takes_root.exists():
        print(f"Cleaning up: Removing {takes_root}")
        shutil.rmtree(takes_root)
    
    duration = (datetime.now() - take_start).total_seconds()
    print(f"\nTake processed in {format_duration(duration)}")
    
    return True


def main():
    """Main execution function."""
    script_start = datetime.now()
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"EgoExo4D Data Download and Processing")
    print(f"Scenario: {args.scenario}")
    print(f"Started: {script_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Load splits
    try:
        splits = load_split_uids(args.scenario)
    except Exception as e:
        print(f"Error loading splits: {e}")
        return 1
    
    # Flatten UIDs
    all_uids = [uid for uids in splits.values() for uid in uids]
    print(f"\nTotal UIDs to process: {len(all_uids)}\n")
    
    output_dir = Path(f"../output_dir_{args.scenario}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process UIDs
    success_count = 0
    failed_uids = []
    uid_times = []
    
    for i, uid in enumerate(all_uids, 1):
        uid_start = datetime.now()
        
        print(f"\n{'*'*80}")
        print(f"Progress: {i}/{len(all_uids)} | Elapsed: {format_duration((uid_start - script_start).total_seconds())}")
        print(f"{'*'*80}")
        
        # Find annotation
        result = find_annotation_for_uid(uid, args.annotation_root)
        if result is None:
            print(f"Annotation not found for {uid}")
            failed_uids.append((uid, "annotation_not_found"))
            continue
        
        annotation, split_name = result
        print(f"Found in {split_name} split")
        
        # Process
        if process_take(uid, annotation, output_dir):
            success_count += 1
            uid_duration = (datetime.now() - uid_start).total_seconds()
            uid_times.append(uid_duration)
            
            # Estimate remaining time
            if len(uid_times) > 0:
                avg_time = sum(uid_times) / len(uid_times)
                remaining = (len(all_uids) - i) * avg_time
                print(f"\nEstimated time remaining: {format_duration(remaining)}")
        else:
            failed_uids.append((uid, "processing_failed"))
    
    # Summary
    total_duration = (datetime.now() - script_start).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {format_duration(total_duration)}")
    print(f"Total UIDs: {len(all_uids)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_uids)}")
    
    if uid_times:
        avg_time = sum(uid_times) / len(uid_times)
        print(f"Average time per UID: {format_duration(avg_time)}")
    
    if failed_uids:
        print(f"\nFailed UIDs:")
        for uid, reason in failed_uids:
            print(f"  {uid}: {reason}")
    
    return 0 if len(failed_uids) == 0 else 1


if __name__ == "__main__":
    exit(main())