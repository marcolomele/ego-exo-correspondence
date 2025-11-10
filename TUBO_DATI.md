# Data Processing Pipeline for Ego-Exo Correspondence

This document describes the two-stage data processing pipeline that prepares EgoExo4D dataset for training correspondence models.

## Pipeline Overview

```
Raw EgoExo4D Dataset
    ↓
[1] download_and_process_data.py
    ↓
Extracted frames + Decoded masks
    ↓
[2] create_pairs.py
    ↓
Training pairs (ego-exo correspondences)
```

---

## Stage 1: `download_and_process_data.py`

### Purpose
Downloads video takes, extracts annotated frames, and prepares mask annotations for efficient runtime loading.

### Input
- **Split files**: `output_dir_{scenario}/split.json` containing train/val/test UIDs
- **Annotation files**: `annotations/relation_annotations/relations_{split}.json` with mask annotations

### Process

**1. Video Download**
```bash
egoexo --parts downscaled_takes/448 --uids {uid}
```
Downloads 448p downscaled videos using the EgoExo4D CLI tool. Changing to `--parts takes` downloads takes in maximum quality. 

**2. Frame Extraction**
- Identifies annotated frames from `annotated_frames` field in annotations
- Extracts only frames with mask annotations (sparse sampling)
- Applies differential downscaling:
  - **Ego cameras** (Aria): ÷2 (preserves detail from first-person view)
  - **Exo cameras**: ÷4 (balances storage vs. quality)
- Saves as JPEG: `{take_uid}/{camera_id}/{frame_idx}.jpg`

**3. Mask Decoding**
- Converts LZString-compressed masks → COCO RLE format
- See "COCO RLE Format" section below for details
- Stores decoded masks in `annotation.json` (not as image files)

**4. Cleanup**
- Deletes original downloaded videos to save disk space
- Retains only extracted frames and processed annotations

### Output Structure
```
output_dir_{scenario}/
├── split.json
└── {take_uid}/
    ├── annotation.json          # Contains masks, object_masks, subsample_idx
    ├── {ego_cam}/               # e.g., aria01_214-1
    │   ├── 0.jpg
    │   ├── 30.jpg
    │   └── ...
    └── {exo_cam}/               # e.g., cam01, cam02
        ├── 0.jpg
        ├── 30.jpg
        └── ...
```

### `annotation.json` Structure
```json
{
  "scenario": "Cooking",
  "take_name": "cmu_cooking_01_...",
  "object_masks": {
    "object_name": {
      "camera_name": {
        "annotation": {
          "frame_id": {
            "width": 3840,
            "height": 2160,
            "encodedMask": "LZString_compressed..."
          }
        },
        "annotated_frames": [0, 30, 60, ...]
      }
    }
  },
  "masks": {
    "object_name": {
      "camera_name": {
        "frame_id": {
          "size": [2160, 3840],
          "counts": "COCO_RLE_string"
        }
      }
    }
  },
  "subsample_idx": [0, 30, 60, 90, ...]
}
```

### Field Distinction: `object_masks` vs `masks`

The `annotation.json` contains two mask representations serving different purposes:

**`object_masks`** - Original annotation data (preserved from EgoExo4D):
- Contains **LZString-compressed** masks (`encodedMask` field)
- Includes full annotation metadata:
  - `width` and `height`: Original frame dimensions
  - `intSegClicks`: Interactive segmentation click points used during annotation
  - `annotated_frames`: List of frame indices with annotations
- **Purpose**: Maintains complete annotation provenance and metadata
- **Not used during training** (too slow to decode)

**`masks`** - Processed masks for efficient training:
- Contains **COCO RLE format** masks (`size` and `counts` fields)
- Decoded from LZString during preprocessing (Stage 1)
- Minimal structure: just the mask data needed for training
- **Purpose**: Fast runtime loading with `pycocotools.mask.decode()`
- **Used by dataloader** during training/evaluation

**Why keep both?**
1. **Reproducibility**: `object_masks` preserves original annotations
2. **Efficiency**: `masks` enables fast training without repeated decoding
3. **Debugging**: Original metadata available if issues arise
4. **Storage overhead**: Minimal (~10-20% increase) due to RLE compression

---

## Stage 2: `create_pairs.py`

### Purpose
Generates training pairs that establish ego-exo correspondences for the same object across viewpoints.

### Input
- Processed data from Stage 1
- `split.json` with train/val/test UIDs

### Process

**1. Iterate through takes**
- For each take in the split, load `annotation.json`

**2. Find ego-exo camera pairs**
- Identify ego camera (contains 'aria' in name)
- Pair with all exo cameras for the same take

**3. Generate correspondence pairs**
Two settings supported:
- **exoego**: exo → ego (find exo point in ego view) <----- our focus for the project.
  - Iterates over exo camera frames
  - For each exo frame, creates pair with corresponding ego frame
- **egoexo**: ego → exo (find ego point in exo view)
  - Iterates over ego camera frames
  - For each ego frame, creates pair with corresponding exo frame

**4. Create virtual paths**
Pair format: `[ego_rgb, ego_mask, exo_rgb, exo_mask]`

Virtual path structure:
```
data_dir//take_uid//camera_id//object_name//rgb//frame_idx
data_dir//take_uid//camera_id//object_name//mask//frame_idx
```

These paths are logical identifiers parsed by the dataloader to:
- Load RGB: `data_dir/take_uid/camera_id/frame_idx.jpg`
- Load mask: `annotation.json['masks'][object_name][camera_id][frame_idx]`

### Output
```
output_dir_{scenario}/
├── train_egoexo_pairs.json
├── val_egoexo_pairs.json
├── test_egoexo_pairs.json
├── train_exoego_pairs.json
├── val_exoego_pairs.json
└── test_exoego_pairs.json
```

Each JSON contains list of 4-tuples:
```json
[
  [
    "output_dir_cooking//take_uid//aria01//object//rgb//0",
    "output_dir_cooking//take_uid//aria01//object//mask//0",
    "output_dir_cooking//take_uid//cam01//object//rgb//0",
    "output_dir_cooking//take_uid//cam01//object//mask//0"
  ],
  ...
]
```

---

## Technical Concepts

### COCO RLE Format

**Purpose**: Efficient storage and manipulation of binary segmentation masks.

**Representation**: Run-Length Encoding (RLE) compresses binary masks by storing runs of consecutive pixels rather than individual pixel values.

**Structure**:
```json
{
  "size": [height, width],
  "counts": "compressed_string"
}
```

**Example**:
- Original binary mask: `[0,0,0,1,1,1,1,0,0,0]` (10 pixels)
- RLE encoding: `[3, 4, 3]` (3 zeros, 4 ones, 3 zeros)
- COCO RLE: Base64-encoded compact representation

**Advantages**:
1. **Storage efficiency**: 10-100x compression vs. raw binary arrays
2. **Fast decoding**: `pycocotools.mask.decode()` is optimized C code
3. **Standard format**: Compatible with COCO ecosystem and evaluation tools
4. **JSON-friendly**: String representation embeds cleanly in JSON

**Usage in pipeline**:
- **Storage**: Masks stored as COCO RLE in `annotation.json`
- **Runtime**: Dataloader calls `mask_utils.decode(rle_obj)` to get numpy binary array
- **No disk I/O**: Eliminates need for separate mask image files

### LZString Compression

Original EgoExo4D annotations use LZString compression (JavaScript library) for masks. The pipeline converts:
```
LZString compressed → Decompressed → COCO RLE → Stored in JSON
```

At training time:
```
JSON → COCO RLE → pycocotools decode → Binary numpy array
```

### Frame Index Convention

- Frames named by original video index (e.g., `0.jpg`, `30.jpg`, `60.jpg`)
- Corresponds to 1 FPS sampling from 30 FPS video
- Frame index serves as temporal alignment key across cameras
- Same index = same timestamp = correspondence constraint

---

## Usage

### Processing Data
```bash
# Process cooking scenario
python download_and_process_data.py --scenario cooking

# Process health scenario
python download_and_process_data.py --scenario health
```

### Creating Training Pairs
```bash
# Generate pairs for cooking
python create_pairs.py --scenario cooking

# Generate pairs for health
python create_pairs.py --scenario health
```

---

## Performance Characteristics

### Storage Efficiency
- **Original videos**: ~2-5 GB per take
- **Processed data**: ~50-200 MB per take (20-40x reduction)
- **Masks**: ~1-5 KB per frame in COCO RLE vs. ~100-500 KB as PNG

### Processing Time
- **Per take**: ~30-60 seconds (download + extraction + mask decoding)
- **Bottleneck**: Network download speed
- **Mask decoding**: ~1-5 ms per mask (LZString → COCO RLE)

### Runtime Performance
- **Mask loading**: ~0.1 ms per mask (COCO RLE → numpy array)
- **No disk I/O** for masks during training (loaded from JSON in memory)
- **Dataloader**: Efficient batch preparation for correspondence training

---

## Data Validation

Processed data is ready for training when:
1. ✅ `split.json` exists in output directory
2. ✅ Each take directory contains `annotation.json` with `masks` field
3. ✅ Frame files exist at `{take_uid}/{camera_id}/{frame_idx}.jpg`
4. ✅ Pair files generated for all splits and settings
5. ✅ Masks in COCO RLE format (not LZString or raw binary)

