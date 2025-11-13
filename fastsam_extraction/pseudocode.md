```python
# High-level pseudocode for FASTSAM mask extraction script

import os
import numpy as np
from fastsam import FastSAM  # From FastSAM repository
import cv2

# Configuration
DATASET_ROOT = "path/to/Ego-Exo4d"
PROCESSED_DIR = os.path.join(DATASET_ROOT, "processed")
SPLITS = ["TRAIN", "VAL", "TEST"]
DIRECTIONS = ["EGO2EXO", "EXO2EGO"]

# Initialize FastSAM model
model = FastSAM(checkpoint_path="path/to/fastsam/weights")

for split in SPLITS:
    for direction in DIRECTIONS:
        # Create output directory: Masks_TRAIN_EGO2EXO, etc.
        output_dir = os.path.join(DATASET_ROOT, f"Masks_{split}_{direction}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate through all takes in the split
        for take_id in get_takes_for_split(split):
            take_dir = os.path.join(PROCESSED_DIR, take_id)
            
            # Iterate through all cameras in the take
            for cam in os.listdir(take_dir):
                cam_dir = os.path.join(take_dir, cam)
                cam_output_dir = os.path.join(output_dir, take_id, cam)
                os.makedirs(cam_output_dir, exist_ok=True)
                
                # Iterate through all frames (*.jpg files)
                for frame_file in sorted(os.listdir(cam_dir)):
                    if not frame_file.endswith('.jpg'):
                        continue
                    
                    frame_idx = frame_file.split('.')[0]  # e.g., "6240"
                    frame_path = os.path.join(cam_dir, frame_file)
                    
                    # Load the frame
                    image = cv2.imread(frame_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Run FastSAM to extract all object masks
                    results = model(
                        image_rgb,
                        device='cuda',
                        retina_masks=True,
                        imgsz=1024,  # Adjust based on your needs
                        conf=0.4,     # Confidence threshold
                        iou=0.9       # NMS IoU threshold
                    )
                    
                    # Extract masks and bounding boxes
                    # Format: masks should be (N, H, W) numpy array
                    # Format: boxes should be (N, 4) numpy array with [x1, y1, w, h]
                    masks = []
                    boxes = []
                    
                    for detection in results:
                        # Extract binary mask (H, W)
                        mask = detection.mask  # Shape: (H, W)
                        masks.append(mask)
                        
                        # Extract bounding box in [x1, y1, w, h] format
                        bbox = detection.box  # Usually in [x1, y1, x2, y2] format
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        boxes.append([x1, y1, w, h])
                    
                    # Convert to numpy arrays
                    masks_array = np.stack(masks, axis=0).astype(np.uint8)  # (N, H, W)
                    boxes_array = np.array(boxes, dtype=np.float32)          # (N, 4)
                    
                    # Save masks as compressed .npz file
                    masks_path = os.path.join(cam_output_dir, f"{frame_idx}_masks.npz")
                    np.savez_compressed(masks_path, masks_array)
                    
                    # Save bounding boxes as .npy file
                    boxes_path = os.path.join(cam_output_dir, f"{frame_idx}_boxes.npy")
                    np.save(boxes_path, boxes_array)
                    
                    print(f"Processed {take_id}/{cam}/{frame_idx}: {len(masks)} masks")

def get_takes_for_split(split):
    """Load the split.json to get takes for train/val/test"""
    with open(os.path.join(PROCESSED_DIR, 'split.json'), 'r') as f:
        splits = json.load(f)
    return splits[split.lower()]
```