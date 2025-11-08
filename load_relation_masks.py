"""
Helper script to load and decode relation masks from EgoExo4D annotations.

This script demonstrates how to:
1. Load relation annotation JSON files
2. Navigate the annotation structure
3. Decode encoded masks to binary numpy arrays
4. Extract mask information for specific takes, objects, and cameras

Usage:
    python load_relation_masks.py <annotation_path> [take_uid] [object_name] [camera_name]
"""

import json
import sys
import numpy as np
from pathlib import Path

try:
    from ego4d.research.util.masks import decode_mask
    USE_EGO4D_UTILS = True
except ImportError:
    USE_EGO4D_UTILS = False
    try:
        from lzstring import LZString
        from pycocotools import mask as mask_utils
    except ImportError:
        print("Warning: Neither ego4d.research.util.masks nor manual decoding dependencies available.")
        print("Install with: pip install lzstring pycocotools")
        sys.exit(1)


def decode_mask_manual(annotation_obj):
    """
    Manually decode encodedMask from annotation object.
    
    Args:
        annotation_obj: Dict with 'width', 'height', 'encodedMask' keys
    
    Returns:
        numpy array: Binary mask (height, width) with 0s and 1s
    """
    width = annotation_obj["width"]
    height = annotation_obj["height"]
    encoded_mask = annotation_obj["encodedMask"]
    
    # Decompress using LZString
    decomp_string = LZString.decompressFromEncodedURIComponent(encoded_mask)
    decomp_encoded = decomp_string.encode()
    
    # Create COCO RLE object
    rle_obj = {
        "size": [height, width],
        "counts": decomp_encoded.decode('ascii')
    }
    
    # Decode RLE to binary mask
    binary_mask = mask_utils.decode(rle_obj)
    return binary_mask


def load_annotations(annotation_path):
    """Load relation annotation JSON file."""
    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    with open(annotation_path, "r") as f:
        relation_ann = json.load(f)
    
    return relation_ann.get("annotations", relation_ann)


def get_takes_with_masks(annotations):
    """Get all take UIDs that have object masks."""
    return {
        take_uid: ann 
        for take_uid, ann in annotations.items() 
        if len(ann.get("object_masks", {})) > 0
    }


def list_available_data(annotations, take_uid=None):
    """Print available takes, objects, and cameras."""
    if take_uid:
        if take_uid not in annotations:
            print(f"Take {take_uid} not found in annotations")
            return
        
        annotation = annotations[take_uid]
        print(f"\nTake: {take_uid}")
        print(f"  Scenario: {annotation.get('scenario', 'N/A')}")
        print(f"  Take Name: {annotation.get('take_name', 'N/A')}")
        
        object_masks = annotation.get("object_masks", {})
        print(f"  Objects: {len(object_masks)}")
        for obj_name in object_masks.keys():
            cameras = list(object_masks[obj_name].keys())
            ego_cams = [c for c in cameras if c.startswith("aria")]
            exo_cams = [c for c in cameras if not c.startswith("aria")]
            print(f"    {obj_name}:")
            print(f"      Ego cameras: {ego_cams}")
            print(f"      Exo cameras: {exo_cams}")
    else:
        relation_takes = get_takes_with_masks(annotations)
        print(f"\nTotal takes with masks: {len(relation_takes)}")
        print("\nFirst 10 takes:")
        for i, take_uid in enumerate(list(relation_takes.keys())[:10]):
            ann = annotations[take_uid]
            num_objects = len(ann.get("object_masks", {}))
            print(f"  {take_uid}: {num_objects} objects - {ann.get('scenario', 'N/A')}")


def get_mask(annotations, take_uid, object_name, camera_name, frame_number):
    """
    Get decoded mask for specific take, object, camera, and frame.
    
    Args:
        annotations: Loaded annotations dict
        take_uid: Take UID string
        object_name: Object name (e.g., "stainless_bowl_0")
        camera_name: Camera name (e.g., "aria01_1201-1" or "cam01")
        frame_number: Frame number as string (e.g., "0", "30", "60")
    
    Returns:
        numpy array: Binary mask (height, width)
    """
    # Navigate to mask annotation
    try:
        mask_annotation = (
            annotations[take_uid]
            ["object_masks"][object_name]
            [camera_name]
            ["annotation"][frame_number]
        )
    except KeyError as e:
        raise KeyError(f"Mask not found: {e}")
    
    # Decode mask
    if USE_EGO4D_UTILS:
        mask = decode_mask(mask_annotation)
    else:
        mask = decode_mask_manual(mask_annotation)
    
    return mask


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample usage:")
        print("  python load_relation_masks.py outdir/annotations/relations_train.json")
        print("  python load_relation_masks.py outdir/annotations/relations_train.json <take_uid>")
        print("  python load_relation_masks.py outdir/annotations/relations_train.json <take_uid> <object_name> <camera_name>")
        sys.exit(1)
    
    annotation_path = sys.argv[1]
    
    # Load annotations
    print(f"Loading annotations from: {annotation_path}")
    annotations = load_annotations(annotation_path)
    print(f"Loaded {len(annotations)} takes")
    
    # List available data
    if len(sys.argv) == 2:
        # Just list all available takes
        list_available_data(annotations)
    
    elif len(sys.argv) == 3:
        # List details for specific take
        take_uid = sys.argv[2]
        list_available_data(annotations, take_uid)
    
    elif len(sys.argv) >= 4:
        # Get specific mask
        take_uid = sys.argv[2]
        object_name = sys.argv[3]
        camera_name = sys.argv[4] if len(sys.argv) > 4 else None
        
        # Get available frames for this object/camera
        annotation = annotations[take_uid]
        object_masks = annotation["object_masks"][object_name]
        
        if camera_name:
            mask_annotations = object_masks[camera_name]["annotation"]
            annotated_frames = object_masks[camera_name]["annotated_frames"]
            
            print(f"\nTake: {take_uid}")
            print(f"Object: {object_name}")
            print(f"Camera: {camera_name}")
            print(f"Annotated frames: {annotated_frames[:10]}...")  # Show first 10
            
            # Get mask for first frame
            frame_number = annotated_frames[0]
            print(f"\nDecoding mask for frame {frame_number}...")
            
            mask = get_mask(annotations, take_uid, object_name, camera_name, str(frame_number))
            
            print(f"Mask shape: {mask.shape}")
            print(f"Mask dtype: {mask.dtype}")
            print(f"Unique values: {np.unique(mask)}")
            print(f"Mask sum (foreground pixels): {mask.sum()}")
            print(f"Mask coverage: {mask.sum() / mask.size * 100:.2f}%")
        else:
            # List available cameras for this object
            cameras = list(object_masks.keys())
            print(f"\nAvailable cameras for object '{object_name}':")
            for cam in cameras:
                num_frames = len(object_masks[cam]["annotation"])
                cam_type = "Ego" if cam.startswith("aria") else "Exo"
                print(f"  {cam} ({cam_type}): {num_frames} annotated frames")


if __name__ == "__main__":
    main()

