# How Relation Data (Masks) Are Obtained from EgoExo4D

This guide explains how to obtain and process relation annotations (segmentation masks) from the EgoExo4D dataset.

## Overview

Relation annotations contain **segmentation masks** that mark the same objects across egocentric and exocentric views. These masks are stored in JSON annotation files that need to be downloaded and decoded.

## Step 1: Download Relation Annotations

Relation annotations are downloaded using the `egoexo` CLI tool. They are part of the "annotations" part and specifically belong to the "relations" benchmark.

### Download Command

```bash
# Download annotations for correspondence/relations benchmark
egoexo -o outdir \
  --parts annotations \
  --benchmarks correspondence \
  --splits train val \
  -y
```

This will download JSON files like:
- `outdir/annotations/relations_train.json`
- `outdir/annotations/relations_val.json`

### Alternative: Download by Take UIDs

If you already have specific take UIDs:

```bash
egoexo -o outdir \
  --parts annotations \
  --uids <take_uid_1> <take_uid_2> \
  -y
```

## Step 2: Understanding the Annotation File Structure

The annotation JSON files have the following structure (using placeholders):

```json
{
  "annotations": {
    "<take_uid>": {
      "scenario": "Making Cucumber & Tomato Salad",
      "take_name": "sfu_cooking_...",
      "object_masks": {
        "<object_name>": {
          "<camera_name>": {
            "annotation": {
              "<frame_number>": {
                "width": 3840,
                "height": 2160,
                "encodedMask": "<LZString_compressed_mask>",
                "intSegClicks": {
                  "positive": [...],
                  "negative": [...]
                }
              }
            },
            "annotation_metadata": {...},
            "annotation_fps": 1,
            "annotated_frames": [0, 30, 60, 90, ...]
          }
        }
      }
    }
  }
}
```

### Key Fields Explained:

- **`take_uid`**: Unique identifier for the video take
- **`object_masks`**: Dictionary keyed by object name (e.g., "stainless_bowl_0")
- **`camera_name`**: 
  - If starts with "aria" → egocentric (ego) camera
  - Otherwise → exocentric (exo) camera (e.g., "cam01", "cam02")
- **`frame_number`**: Frame index in the original video (typically sampled at 1fps: 0, 30, 60, ...)
- **`encodedMask`**: LZString-compressed mask string that needs to be decoded
- **`width`/`height`**: Dimensions of the video frame (needed for decoding)

## Step 3: Decoding the Masks

The `encodedMask` field contains a compressed string that needs to be decoded into a binary mask. Based on the [EgoExo Relations notebook](https://github.com/facebookresearch/Ego4d/blob/main/notebooks/egoexo/EgoExo_Relations.ipynb), you can use the `decode_mask` function from the Ego4D research utilities.

### Method 1: Using Ego4D Research Utilities

```python
from ego4d.research.util.masks import decode_mask
import json

# Load annotation file
with open("outdir/annotations/relations_train.json", "r") as f:
    relation_ann = json.load(f)

annotations = relation_ann["annotations"]

# Get a specific take
take_uid = "<take_uid_here>"
annotation = annotations[take_uid]

# Get masks for a specific object and camera
object_name = "<object_name>"  # e.g., "stainless_bowl_0"
camera_name = "<camera_name>"  # e.g., "aria01_1201-1" or "cam01"
frame_number = "0"  # Frame number as string

# Extract mask annotation
mask_annotation = annotation["object_masks"][object_name][camera_name]["annotation"][frame_number]

# Decode the mask
mask = decode_mask(mask_annotation)
# Returns: numpy array of shape (height, width) with binary values (0 or 1)
```

### Method 2: Manual Decoding (Alternative)

If you don't have access to Ego4D utilities, you can decode manually using the same approach as `process_data.py`:

```python
from lzstring import LZString
from pycocotools import mask as mask_utils

def decode_mask_manual(annotation_obj):
    """
    Decode encodedMask from annotation object.
    
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
```

## Step 4: Visualizing Masks

Based on the notebook, you can visualize masks by blending them with the video frames:

```python
from ego4d.research.util.masks import blend_mask
from PIL import Image
import numpy as np

# Load video frame (using your video reader)
# frame = reader[frame_number]  # Get frame from video
# input_img = frame["video"][0].numpy()

# Decode mask
mask = decode_mask(mask_annotation)

# Blend mask with image (alpha controls transparency)
blended = blend_mask(input_img, mask, alpha=0.7)

# Display
pil_img = Image.fromarray(blended)
pil_img.show()
```

## Step 5: Finding Available Data

To find which takes have relation annotations:

```python
import json

# Load annotation file
with open("outdir/annotations/relations_train.json", "r") as f:
    relation_ann = json.load(f)

annotations = relation_ann["annotations"]

# Get all takes with masks
relation_takes = {
    take_uid: ann 
    for take_uid, ann in annotations.items() 
    if len(ann.get("object_masks", {})) > 0
}

print(f"Found {len(relation_takes)} takes with relation annotations")

# Get all objects for a take
take_uid = list(relation_takes.keys())[0]
object_masks = annotations[take_uid]["object_masks"]
print(f"Objects in take: {list(object_masks.keys())}")

# Get all cameras for an object
object_name = list(object_masks.keys())[0]
cameras = list(object_masks[object_name].keys())
print(f"Cameras with annotations: {cameras}")

# Get all annotated frames
camera_name = cameras[0]
annotated_frames = object_masks[object_name][camera_name]["annotated_frames"]
print(f"Annotated frames: {annotated_frames[:10]}...")  # First 10
```

## Complete Example Workflow

```python
import json
import os
from ego4d.research.util.masks import decode_mask, blend_mask
from PIL import Image

# 1. Load annotations
annotation_path = "outdir/annotations/relations_train.json"
with open(annotation_path, "r") as f:
    relation_ann = json.load(f)

annotations = relation_ann["annotations"]

# 2. Select a take with masks
relation_takes = {
    k: v for k, v in annotations.items() 
    if len(v.get("object_masks", {})) > 0
}
take_uid = list(relation_takes.keys())[0]
print(f"Processing take: {take_uid}")

# 3. Get object masks
annotation = annotations[take_uid]
object_masks = annotation["object_masks"]

# 4. Select object and camera
object_name = list(object_masks.keys())[0]
camera_name = list(object_masks[object_name].keys())[0]
print(f"Object: {object_name}, Camera: {camera_name}")

# 5. Get a frame
mask_annotations = object_masks[object_name][camera_name]["annotation"]
frame_number = list(mask_annotations.keys())[0]
print(f"Frame: {frame_number}")

# 6. Decode mask
mask_annotation = mask_annotations[frame_number]
mask = decode_mask(mask_annotation)
print(f"Mask shape: {mask.shape}")
print(f"Mask values: {np.unique(mask)}")  # Should be [0, 1] or [0, 255]

# 7. (Optional) Load corresponding video frame and visualize
# This requires loading the video file separately
# video_path = get_video_path(take_uid, camera_name)
# frame = load_frame(video_path, int(frame_number))
# blended = blend_mask(frame, mask, alpha=0.7)
# Image.fromarray(blended).show()
```

## Summary

1. **Download**: Use `egoexo` CLI to download relation annotations
2. **Location**: JSON files in `outdir/annotations/relations_*.json`
3. **Structure**: Nested dictionaries: `take_uid → object_name → camera_name → frame_number`
4. **Decoding**: Use `decode_mask()` from Ego4D utilities or manual LZString + COCO RLE decoding
5. **Output**: Binary numpy arrays (height × width) with 0s (background) and 1s (object)

## References

- [EgoExo Relations Notebook](https://github.com/facebookresearch/Ego4d/blob/main/notebooks/egoexo/EgoExo_Relations.ipynb)
- [Relations Annotations Documentation](https://docs.ego-exo4d-data.org/annotations/relations/)
- [SegSwap process_data.py](https://github.com/EGO4D/ego-exo4d-relation/blob/main/correspondence/SegSwap/data/process_data.py)

