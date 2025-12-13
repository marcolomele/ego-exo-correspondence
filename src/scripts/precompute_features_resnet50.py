"""
Pre-extract ResNet-50 (DINO pretrained) features for a random sample of frames targeting ~13K ego-exo pairs.
Run this script once before training to cache features on disk.

This script:
1. Loads all pairs from train/val/test JSON files
2. Randomly samples frames to get approximately 13K ego-exo frame pairs
3. Collects all associated masks (masks are included in pair structure)
4. Precomputes features only for frames in the sampled pairs
5. Saves new JSON files with sampled pairs (backing up originals)
6. Creates a new split.json with only takes that have sampled pairs

Usage:
    python precompute_features_resnet50.py --root ../../data/health_normal_data_omama --stride 32

Output structure:
    {root}/precomputed_features/{take_id}/{cam}/{idx}.npz
    {root}/dataset_jsons/{train/val/test}_exoego_pairs.json (sampled, originals backed up)
    {root}/processed/split.json (new split with only sampled takes, original backed up)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms as transforms
from collections import defaultdict
import random


def resize_image(img, target_h, target_w):
    """Resize image with appropriate interpolation method."""
    h, w = img.shape[:2]
    if target_h > h or target_w > w:
        interpolation = cv2.INTER_LINEAR  # Better for upscaling
    else:
        interpolation = cv2.INTER_AREA  # Better for downscaling
    return cv2.resize(img, (target_w, target_h), interpolation=interpolation)


def get_target_size(reverse, is_source):
    """
    Get target image size to preserve aspect ratios and match DINOv2 feature grids.
    
    These sizes are then adaptively pooled to 50×50 or 38×68 to match DINOv2 grids.
    """
    if reverse:
        if is_source:
            return 608, 1088  # exo camera (rectangular)
        else:
            return 800, 800   # ego camera (square)
    else:
        if is_source:
            return 800, 800   # ego camera (square)
        else:
            return 608, 1088  # exo camera (rectangular)


def get_target_pool_size(reverse, is_source):
    if reverse:
        if is_source:
            return 38, 68  # exo → 38×68 grid
        else:
            return 50, 50  # ego → 50×50 grid
    else:
        if is_source:
            return 50, 50  # ego → 50×50 grid
        else:
            return 38, 68  # exo → 38×68 grid


class ResNetMultiLayer(nn.Module):
    """ResNet-50 backbone pretrained with DINO v1."""
    
    def __init__(self):
        super().__init__()
        # Load DINO v1 pretrained ResNet-50 backbone from the original DINO repository
        self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
        # Remove the final FC layer to get feature maps
        self.resnet.fc = nn.Identity()
        self.resnet.eval()

    def forward(self, x, target_pool_size=None):
        # Extract features without the classification head
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        features = x
        
        # Adaptive pooling to match DINOv2 feature grids (50×50 or 38×68)
        if target_pool_size is not None:
            features = F.adaptive_avg_pool2d(features, target_pool_size)
        else:
            # Default: assume square 50×50 for backward compatibility
            features = F.adaptive_avg_pool2d(features, (50, 50))
        
        return features


class DINOFeatureExtractor:
    """Extract dense features from images using ResNet-50 pretrained with DINO."""
    
    def __init__(self, device):
        # Load ResNet-50 with DINO pretraining
        print(f"Loading DINO ResNet-50 model")
        self.model = ResNetMultiLayer().to(device)
        self.model.eval()
        
        self.device = device
        self.feat_dim = 2048  # Raw ResNet-50 features (before projection)
        
        print(f"Feature dimension: {self.feat_dim}")
        print(f"Model stride: 32 (fixed for ResNet-50)")
        print(f"Features adaptively pooled to 50×50 (ego) or 38×68 (exo) grids")
        
        # Image transforms (same as in dataset_masks.py)
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(norm_mean, norm_std)
        ])
    
    @torch.no_grad()
    def extract(self, image_tensor, target_pool_size=None):
        """
        Extract dense features from image tensor.
        
        Args:
            image_tensor: [B, 3, H, W] normalized image tensor
            target_pool_size: (H, W) tuple for adaptive pooling (e.g., (50, 50) or (38, 68))
            
        Returns:
            features: [B, C, H_pool, W_pool] feature tensor
        """
        features = self.model(image_tensor.to(self.device), target_pool_size=target_pool_size)
        return features.cpu()


def load_all_pairs(root, reverse):
    """
    Load all pairs from train/val/test JSON files.
    Always loads exoego pairs regardless of reverse flag.
    
    Returns:
        dict: {'train': list, 'val': list, 'test': list} of pairs
    """
    jsons_dir = Path(root) / 'dataset_jsons'
    
    all_pairs = {'train': [], 'val': [], 'test': []}
    
    # Always load exoego pairs
    json_files = {
        'train': 'train_exoego_pairs.json',
        'val': 'val_exoego_pairs.json',
        'test': 'test_exoego_pairs.json'
    }
    
    for split, json_file in json_files.items():
        json_path = jsons_dir / json_file
        if not json_path.exists():
            print(f"Skipping {json_file} (not found)")
            continue
            
        print(f"Loading {json_path.name}...")
        with open(json_path) as f:
            pairs = json.load(f)
        all_pairs[split] = pairs
        print(f"  Loaded {len(pairs)} pairs from {split}")
    
    return all_pairs


def sample_frames(all_pairs, target_pairs=13000, seed=42):
    """
    Randomly sample frames to get approximately target_pairs ego-exo frame pairs.
    Masks are automatically included as they're part of the pair structure.
    
    Args:
        all_pairs: dict with 'train', 'val', 'test' keys containing lists of pairs
        target_pairs: target number of pairs to sample (default: 13000)
        seed: random seed for reproducibility
    
    Returns:
        dict: sampled pairs with same structure as all_pairs
    """
    print("Counting total pairs...")
    random.seed(seed)
    np.random.seed(seed)
    
    # Count total pairs across all splits
    total_pairs = sum(len(all_pairs[split]) for split in ['train', 'val', 'test'])
    
    print(f"\nTotal pairs available: {total_pairs}")
    print(f"Target pairs: {target_pairs}")
    
    if target_pairs >= total_pairs:
        print(f"Target ({target_pairs}) >= total pairs ({total_pairs}), using all pairs")
        return all_pairs
    
    # Calculate sample ratio needed
    sample_ratio = target_pairs / total_pairs
    
    # First, collect all unique frames from all pairs
    print("Collecting unique frames (this is slow for large datasets)...")
    unique_frames = set()
    total_to_process = sum(len(all_pairs[split]) for split in ['train', 'val', 'test'])
    processed_count = 0
    
    for split in ['train', 'val', 'test']:
        for pair in all_pairs[split]:
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"  Processed {processed_count}/{total_to_process} pairs...")
            # Pair format: [img1_rgb, img1_mask, img2_rgb, img2_mask]
            img_pth1, _, img_pth2, _ = pair
            # Parse frame identifiers (take_id, cam, idx)
            _, take_id1, cam1, _, _, idx1 = img_pth1.split('//')
            _, take_id2, cam2, _, _, idx2 = img_pth2.split('//')
            unique_frames.add((take_id1, cam1, idx1))
            unique_frames.add((take_id2, cam2, idx2))
    
    total_frames = len(unique_frames)
    # Sample frames with a slightly higher ratio to account for pairs needing both frames
    # Use iterative approach: sample frames, count pairs, adjust if needed
    frame_sample_ratio = min(1.0, sample_ratio * 1.5)  # Start with higher ratio
    num_frame_samples = int(total_frames * frame_sample_ratio)
    
    print(f"\nSampling {num_frame_samples} out of {total_frames} unique frames ({frame_sample_ratio*100:.1f}%)")
    
    # Randomly sample frames
    sampled_frames = set(random.sample(list(unique_frames), num_frame_samples))
    
    # Filter pairs to keep only those where both frames are in the sampled set
    sampled_pairs = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        for pair in all_pairs[split]:
            img_pth1, _, img_pth2, _ = pair
            _, take_id1, cam1, _, _, idx1 = img_pth1.split('//')
            _, take_id2, cam2, _, _, idx2 = img_pth2.split('//')
            
            frame1 = (take_id1, cam1, idx1)
            frame2 = (take_id2, cam2, idx2)
            
            # Keep pair only if both frames are sampled
            if frame1 in sampled_frames and frame2 in sampled_frames:
                sampled_pairs[split].append(pair)
    
    total_sampled = sum(len(sampled_pairs[split]) for split in ['train', 'val', 'test'])
    
    # If we didn't get enough pairs, iteratively add more frames
    max_iterations = 10
    iteration = 0
    while total_sampled < target_pairs and iteration < max_iterations:
        iteration += 1
        remaining_frames = list(unique_frames - sampled_frames)
        if len(remaining_frames) == 0:
            break
        
        # Calculate how many more frames we need
        needed_pairs = target_pairs - total_sampled
        # Estimate: each new frame might add ~0.5 pairs on average (since pairs need 2 frames)
        frames_to_add = min(len(remaining_frames), int(needed_pairs * 2))
        
        additional_frames = set(random.sample(remaining_frames, frames_to_add))
        sampled_frames.update(additional_frames)
        
        # Re-filter pairs with expanded frame set
        sampled_pairs = {'train': [], 'val': [], 'test': []}
        for split in ['train', 'val', 'test']:
            for pair in all_pairs[split]:
                img_pth1, _, img_pth2, _ = pair
                _, take_id1, cam1, _, _, idx1 = img_pth1.split('//')
                _, take_id2, cam2, _, _, idx2 = img_pth2.split('//')
                
                frame1 = (take_id1, cam1, idx1)
                frame2 = (take_id2, cam2, idx2)
                
                if frame1 in sampled_frames and frame2 in sampled_frames:
                    sampled_pairs[split].append(pair)
        
        total_sampled = sum(len(sampled_pairs[split]) for split in ['train', 'val', 'test'])
        
        if total_sampled >= target_pairs:
            break
    
    # If we have too many pairs, randomly sample down to target
    if total_sampled > target_pairs:
        all_sampled_flat = []
        for split in ['train', 'val', 'test']:
            for pair in sampled_pairs[split]:
                all_sampled_flat.append((split, pair))
        
        selected = random.sample(all_sampled_flat, target_pairs)
        sampled_pairs = {'train': [], 'val': [], 'test': []}
        for split, pair in selected:
            sampled_pairs[split].append(pair)
        total_sampled = target_pairs
    
    print(f"\nFinal sampled pairs: {total_sampled}")
    print(f"Sampled pairs per split:")
    print(f"  Train: {len(sampled_pairs['train'])} / {len(all_pairs['train'])}")
    print(f"  Val: {len(sampled_pairs['val'])} / {len(all_pairs['val'])}")
    print(f"  Test: {len(sampled_pairs['test'])} / {len(all_pairs['test'])}")
    print(f"\nNote: All associated masks are included in the sampled pairs.")
    
    return sampled_pairs


def collect_unique_images_from_pairs(sampled_pairs, reverse):
    """
    Collect all unique images from the sampled pairs.
    Always processes exoego pairs: img1=exo, img2=ego.
    The reverse flag determines which is source/dest for feature extraction.
    
    Returns:
        dict: {(take_id, cam, idx): {'is_source': bool, 'is_dest': bool}}
    """
    images = defaultdict(lambda: {'is_source': False, 'is_dest': False})
    
    for split in ['train', 'val', 'test']:
        for pair in sampled_pairs[split]:
            # Always exoego pairs: [img1_rgb, img1_mask, img2_rgb, img2_mask]
            # img1 = exo camera, img2 = ego camera
            img_pth_exo, _, img_pth_ego, _ = pair
            
            # Determine source/dest based on reverse flag
            # reverse=True: exo->ego (exo is source, ego is dest)
            # reverse=False: ego->exo (ego is source, exo is dest)
            if reverse:
                # exo->ego: exo is source, ego is dest
                img_pth_source = img_pth_exo
                img_pth_dest = img_pth_ego
            else:
                # ego->exo: ego is source, exo is dest
                img_pth_source = img_pth_ego
                img_pth_dest = img_pth_exo
            
            # Parse source image path
            _, take_id, cam, obj, _, idx = img_pth_source.split('//')
            images[(take_id, cam, idx)]['is_source'] = True
            
            # Parse dest image path
            _, take_id2, cam2, obj2, _, idx2 = img_pth_dest.split('//')
            images[(take_id2, cam2, idx2)]['is_dest'] = True
    
    return images


def save_sampled_json_files(root, sampled_pairs, reverse):
    """
    Save new JSON files with sampled pairs and create new split.json.
    Always saves exoego pairs regardless of reverse flag.
    
    Args:
        root: dataset root directory
        sampled_pairs: dict with 'train', 'val', 'test' keys containing sampled pairs
        reverse: flag for feature extraction direction (not used for file naming)
    """
    jsons_dir = Path(root) / 'dataset_jsons'
    jsons_dir.mkdir(parents=True, exist_ok=True)
    
    # Always save exoego pairs
    output_files = {
        'train': 'train_exoego_pairs.json',
        'val': 'val_exoego_pairs.json',
        'test': 'test_exoego_pairs.json'
    }
    
    # Backup original files if they exist
    print("\nSaving sampled pairs JSON files...")
    for split in ['train', 'val', 'test']:
        output_path = jsons_dir / output_files[split]
        
        # Backup original if it exists
        if output_path.exists():
            backup_path = jsons_dir / f"{output_files[split]}.backup"
            import shutil
            shutil.copy2(output_path, backup_path)
            print(f"  Backed up original {output_files[split]} to {output_files[split]}.backup")
        
        with open(output_path, 'w') as f:
            json.dump(sampled_pairs[split], f, indent=2)
        print(f"  Saved {output_files[split]} with {len(sampled_pairs[split])} pairs")
    
    # Create new split.json with only takes that have sampled pairs
    print("\nCreating new split.json...")
    takes_in_sampled = {'train': set(), 'val': set(), 'test': set()}
    
    for split in ['train', 'val', 'test']:
        for pair in sampled_pairs[split]:
            # Extract take_id from both source and dest images
            # Pair format: [img1_rgb, img1_mask, img2_rgb, img2_mask]
            img_pth1, _, img_pth2, _ = pair
            
            _, take_id, _, _, _, _ = img_pth1.split('//')
            _, take_id2, _, _, _, _ = img_pth2.split('//')
            
            takes_in_sampled[split].add(take_id)
            takes_in_sampled[split].add(take_id2)
    
    # Convert sets to sorted lists
    new_split = {
        'train': sorted(list(takes_in_sampled['train'])),
        'val': sorted(list(takes_in_sampled['val'])),
        'test': sorted(list(takes_in_sampled['test']))
    }
    
    # Save new split.json in processed directory
    processed_dir = Path(root) / 'processed'
    split_json_path = processed_dir / 'split.json'
    
    # Backup original if it exists
    if split_json_path.exists():
        backup_path = processed_dir / 'split.json.backup'
        import shutil
        shutil.copy2(split_json_path, backup_path)
        print(f"  Backed up original split.json to split.json.backup")
    
    with open(split_json_path, 'w') as f:
        json.dump(new_split, f, indent=2)
    
    print(f"  Saved new split.json:")
    print(f"    Train takes: {len(new_split['train'])}")
    print(f"    Val takes: {len(new_split['val'])}")
    print(f"    Test takes: {len(new_split['test'])}")


def precompute_features(args):
    """Main function to precompute and save ResNet-50 (DINO pretrained) features."""
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Running on CPU will be EXTREMELY slow!")
        print("To use GPU, make sure CUDA is available and use --device 0")
    else:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # Load all pairs from JSON files
    print("\n" + "=" * 60)
    print("LOADING ALL PAIRS FROM DATASET")
    print("=" * 60)
    all_pairs = load_all_pairs(args.root, args.reverse)
    
    # Sample frames to get approximately 13K ego-exo frame pairs
    target_pairs = 13000
    print("\n" + "=" * 60)
    print(f"SAMPLING FRAMES TO GET ~{target_pairs} EGO-EXO PAIRS")
    print("This may take a few minutes for large datasets...")
    print("=" * 60)
    sampled_pairs = sample_frames(all_pairs, target_pairs=target_pairs, seed=args.seed)
    
    # Save sampled JSON files and create new split.json
    print("\n" + "=" * 60)
    print("SAVING SAMPLED JSON FILES")
    print("=" * 60)
    save_sampled_json_files(args.root, sampled_pairs, args.reverse)
    
    # Initialize feature extractor
    extractor = DINOFeatureExtractor(device=device)
    
    # Create output directory
    features_dir = Path(args.root) / 'precomputed_features'
    features_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {features_dir}")
    
    # Collect unique images from sampled pairs
    print("\n" + "=" * 60)
    print("COLLECTING UNIQUE IMAGES FROM SAMPLED PAIRS")
    print("=" * 60)
    images = collect_unique_images_from_pairs(sampled_pairs, args.reverse)
    print(f"Found {len(images)} unique images to process")
    
    # Count how many need source/dest features
    source_count = sum(1 for v in images.values() if v['is_source'])
    dest_count = sum(1 for v in images.values() if v['is_dest'])
    print(f"  - Source features needed: {source_count}")
    print(f"  - Dest features needed: {dest_count}")
    
    # Process each image
    dataset_dir = Path(args.root) / 'processed'
    skipped = 0
    processed = 0
    errors = 0
    
    import time
    import gc
    timings = {'imread': 0, 'resize': 0, 'transform': 0, 'extract': 0, 'save': 0}
    iter_count = 0
    
    # BATCH PROCESSING - collect items to process
    batch_items = []
    for (take_id, cam, idx), roles in images.items():
        vid_idx = int(idx)
        img_path = dataset_dir / take_id / cam / f"{vid_idx}.jpg"
        
        if not img_path.exists():
            errors += 1
            continue
        
        out_dir = features_dir / take_id / cam
        out_path_source = out_dir / f"{vid_idx}_source.npz"
        out_path_dest = out_dir / f"{vid_idx}_dest.npz"
        
        process_source = roles['is_source'] and (args.overwrite or not out_path_source.exists())
        process_dest = roles['is_dest'] and (args.overwrite or not out_path_dest.exists())
        
        if not process_source and not process_dest:
            skipped += 2 if (roles['is_source'] and roles['is_dest']) else 1
        else:
            batch_items.append({
                'take_id': take_id, 'cam': cam, 'idx': vid_idx,
                'img_path': img_path, 'out_dir': out_dir,
                'process_source': process_source, 'process_dest': process_dest
            })
    
    print(f"Items to process: {len(batch_items)}")
    print(f"Skipped (already exist): {skipped}")
    
    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(batch_items) + batch_size - 1) // batch_size
    
    print(f"\nProcessing in batches of {batch_size}...")
    
    with torch.no_grad():  # Critical for speed and memory
        for batch_idx in tqdm(range(num_batches), desc="Batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(batch_items))
            current_batch = batch_items[batch_start:batch_end]
            
            # Load and process all images in batch
            source_tensors = []
            source_items = []
            dest_tensors = []
            dest_items = []
            
            t0 = time.time()
            for item in current_batch:
                img = cv2.imread(str(item['img_path']))[..., ::-1].copy()
                
                # Process source and dest separately (different sizes and pool targets)
                if item['process_source']:
                    # Use proper aspect-ratio preserving sizes like DINOv2
                    target_h, target_w = get_target_size(args.reverse, is_source=True)
                    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    
                    img_tensor = extractor.transform(img_resized)
                    source_tensors.append(img_tensor)
                    source_items.append(item)
                
                if item['process_dest']:
                    # Use proper aspect-ratio preserving sizes like DINOv2
                    target_h, target_w = get_target_size(args.reverse, is_source=False)
                    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    
                    img_tensor = extractor.transform(img_resized)
                    dest_tensors.append(img_tensor)
                    dest_items.append(item)
            
            timings['imread'] += time.time() - t0
            
            # Extract source features in batch
            if source_tensors:
                t0 = time.time()
                source_batch = torch.stack(source_tensors)
                # Get target pool size for source (50×50 or 38×68)
                target_pool_size = get_target_pool_size(args.reverse, is_source=True)
                features_batch = extractor.extract(source_batch, target_pool_size=target_pool_size)
                timings['extract'] += time.time() - t0
                
                t0 = time.time()
                for i, item in enumerate(source_items):
                    item['out_dir'].mkdir(parents=True, exist_ok=True)
                    out_path = item['out_dir'] / f"{item['idx']}_source.npz"
                    np.savez_compressed(str(out_path), features=features_batch[i:i+1].numpy())
                    processed += 1
                timings['save'] += time.time() - t0
                del source_batch, features_batch
            
            # Extract dest features in batch
            if dest_tensors:
                t0 = time.time()
                dest_batch = torch.stack(dest_tensors)
                # Get target pool size for dest (50×50 or 38×68)
                target_pool_size = get_target_pool_size(args.reverse, is_source=False)
                features_batch = extractor.extract(dest_batch, target_pool_size=target_pool_size)
                timings['extract'] += time.time() - t0
                
                t0 = time.time()
                for i, item in enumerate(dest_items):
                    item['out_dir'].mkdir(parents=True, exist_ok=True)
                    out_path = item['out_dir'] / f"{item['idx']}_dest.npz"
                    np.savez_compressed(str(out_path), features=features_batch[i:i+1].numpy())
                    processed += 1
                timings['save'] += time.time() - t0
                del dest_batch, features_batch
            
            # Clear memory
            if batch_idx % 10 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Show timing after first batch
            if batch_idx == 0 and processed > 0:
                total_time = sum(timings.values())
                print(f"\n\nFirst batch timing:")
                print(f"  Load+preprocess: {timings['imread']:.2f}s")
                print(f"  GPU extraction:  {timings['extract']:.2f}s")
                print(f"  Save to disk:    {timings['save']:.2f}s")
                print(f"  Time per batch:  {total_time:.2f}s")
                est_total_min = (total_time * num_batches) / 60
                print(f"  Estimated total: {est_total_min:.1f} minutes\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output directory: {features_dir}")
    
    # Print timing breakdown
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN (total seconds)")
    print("=" * 60)
    total_time = sum(timings.values())
    for op, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (t / total_time * 100) if total_time > 0 else 0
        avg = (t / processed * 1000) if processed > 0 else 0
        print(f"{op:12s}: {t:7.2f}s ({pct:5.1f}%) - avg {avg:6.1f}ms per image")
    
    # Estimate storage size
    total_size = sum(f.stat().st_size for f in features_dir.rglob("*.npz"))
    print(f"Total storage used: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-extract ResNet-50 (DINO pretrained) features for O-MaMa dataset")
    parser.add_argument("--root", type=str, required=True,
                       help="Path to the dataset root directory")
    parser.add_argument("--reverse", action="store_true",
                       help="Extract for exo->ego direction (default: ego->exo)")
    parser.add_argument("--device", default="0", type=str,
                       help="GPU device ID or 'cpu'")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing feature files")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed error messages")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (default: 42)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for GPU processing (default: 32, reduce if OOM)")
    
    args = parser.parse_args()
    precompute_features(args)

