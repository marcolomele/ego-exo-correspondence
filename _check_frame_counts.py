#!/usr/bin/env python3
"""
Script to check if any objects have fewer annotated frames in EGO view than EXO view.

For each object under object_masks:
- EGO: cameras with names containing "aria"
- EXO: cameras with names not containing "aria"
- Returns object names where len(EGO annotated_frames) < len(EXO annotated_frames)
"""

import json
import sys
from pathlib import Path


def check_frame_counts(annotation_path):
    """
    Check for objects where EGO annotated_frames < EXO annotated_frames.
    
    Args:
        annotation_path: Path to the JSON annotation file
    
    Returns:
        List of object names that meet the condition
    """
    # Load JSON file
    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    print(f"Loading annotations from: {annotation_path}")
    with open(annotation_path, "r") as f:
        data = json.load(f)
    
    # Get annotations dict (handle both formats)
    annotations = data.get("annotations", data)
    
    objects_with_less_ego_frames = []
    
    # Iterate through each take
    for take_uid, take_data in annotations.items():
        object_masks = take_data.get("object_masks", {})
        
        # Iterate through each object
        for object_name, cameras in object_masks.items():
            # Check each camera for this object
            ego_total_frames = 0
            exo_total_frames = 0
            ego_camera_names = []
            exo_camera_names = []
            
            for camera_name, camera_data in cameras.items():
                annotated_frames = camera_data.get("annotated_frames", [])
                num_frames = len(annotated_frames)
                
                # Classify as EGO or EXO
                if "aria" in camera_name.lower():
                    # EGO camera
                    ego_total_frames += num_frames
                    ego_camera_names.append(camera_name)
                else:
                    # EXO camera
                    exo_total_frames += num_frames
                    exo_camera_names.append(camera_name)
            
            # Compare: if total EGO frames < total EXO frames
            if ego_camera_names and exo_camera_names and ego_total_frames < exo_total_frames:
                objects_with_less_ego_frames.append({
                    "take_uid": take_uid,
                    "object_name": object_name,
                    "ego_total_frames": ego_total_frames,
                    "exo_total_frames": exo_total_frames,
                    "ego_cameras": ego_camera_names,
                    "exo_cameras": exo_camera_names
                })
    
    return objects_with_less_ego_frames


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_frame_counts.py <annotation_json_path>")
        print("Example: python check_frame_counts.py exract_test.json")
        sys.exit(1)
    
    annotation_path = sys.argv[1]
    
    try:
        results = check_frame_counts(annotation_path)
        
        if results:
            print(f"\nFound {len(results)} object(s) with fewer EGO frames than EXO frames:\n")
            for result in results:
                print(f"Take: {result['take_uid']}")
                print(f"Object: {result['object_name']}")
                print(f"  EGO total frames: {result['ego_total_frames']} (cameras: {result['ego_cameras']})")
                print(f"  EXO total frames: {result['exo_total_frames']} (cameras: {result['exo_cameras']})")
                print()
            
            # Return just the object names as requested
            object_names = [r['object_name'] for r in results]
            print(f"\nObject names: {object_names}")
        else:
            print("\nNo objects found with fewer EGO frames than EXO frames.")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

