"""
Pre-extract DINOv2 features for a random sample of frames targeting ~13K ego-exo pairs.
Run this script once before training to cache features on disk.

This script:
1. Loads all pairs from train/val/test JSON files
2. Randomly samples frames to get approximately 13K ego-exo frame pairs
3. Collects all associated masks (masks are included in pair structure)
4. Precomputes features only for frames in the sampled pairs
5. Saves new JSON files with sampled pairs (backing up originals)
6. Creates a new split.json with only takes that have sampled pairs

Usage:
    python precompute_features.py --root ../../data/health_normal_data_omama --patch_size 16

Output structure:
    {root}/precomputed_features/{take_id}/{cam}/{idx}.npz
    {root}/dataset_jsons/{train/val/test}_exoego_pairs.json (sampled, originals backed up)
    {root}/processed/split.json (new split with only sampled takes, original backed up)
"""

import torch
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


def get_target_size(patch_size, reverse, is_source):
    """
    Get target image size based on model position embeddings.
    
    Model expects:
        - reverse mode (exo->ego): source=38*68, dest=50*50
        - normal mode (ego->exo): source=50*50, dest=38*68
    """
    if reverse:
        if is_source:
            return 38 * patch_size, 68 * patch_size  # 608x1088 for patch_size=16
        else:
            return 50 * patch_size, 50 * patch_size  # 800x800 for patch_size=16
    else:
        if is_source:
            return 50 * patch_size, 50 * patch_size
        else:
            return 38 * patch_size, 68 * patch_size


def resize_image(img, target_h, target_w):
    """Resize image with appropriate interpolation method."""
    h, w = img.shape[:2]
    if target_h > h or target_w > w:
        interpolation = cv2.INTER_LINEAR  # Better for upscaling
    else:
        interpolation = cv2.INTER_AREA  # Better for downscaling
    return cv2.resize(img, (target_w, target_h), interpolation=interpolation)


class DINOFeatureExtractor:
    """Extract dense features from images using DINOv2."""
    
    def __init__(self, model_name, patch_size, device):
        # Load DINOv2 pre-trained model
        print(f"Loading DINOv2 model: {model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()
        
        self.patch_size = patch_size
        self.device = device
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            dummy_features = self.model.forward_features(dummy_input)
            self.feat_dim = dummy_features['x_norm_patchtokens'].shape[-1]
        print(f"Feature dimension: {self.feat_dim}")
        
        # Image transforms (same as in dataset_masks.py)
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(norm_mean, norm_std)
        ])
    
    @torch.no_grad()
    def extract(self, image_tensor):
        """
        Extract dense features from image tensor.
        
        Args:
            image_tensor: [B, 3, H, W] normalized image tensor
            
        Returns:
            patch_tokens: [B, C, H//patch_size, W//patch_size] feature tensor
        """
        B, _, h, w = image_tensor.shape
        features = self.model.forward_features(image_tensor.to(self.device))
        patch_tokens = features['x_norm_patchtokens']
        
        # Reshape to [B, C, H, W]
        patch_tokens = patch_tokens.reshape(
            B, h // self.patch_size, w // self.patch_size, -1
        ).permute(0, 3, 1, 2)
        
        return patch_tokens.cpu()


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
    unique_frames = set()
    for split in ['train', 'val', 'test']:
        for pair in all_pairs[split]:
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
            # img1 = ego camera, img2 = exo camera
            img_pth_ego, _, img_pth_exo, _ = pair
                
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
    """Main function to precompute and save DINOv2 features."""
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load all pairs from JSON files
    print("\n" + "=" * 60)
    print("LOADING ALL PAIRS FROM DATASET")
    print("=" * 60)
    all_pairs = load_all_pairs(args.root, args.reverse)
    
    # Sample frames to get approximately 13K ego-exo frame pairs
    target_pairs = 13000
    print("\n" + "=" * 60)
    print(f"SAMPLING FRAMES TO GET ~{target_pairs} EGO-EXO PAIRS")
    print("=" * 60)
    sampled_pairs = sample_frames(all_pairs, target_pairs=target_pairs, seed=args.seed)
    
    # Save sampled JSON files and create new split.json
    print("\n" + "=" * 60)
    print("SAVING SAMPLED JSON FILES")
    print("=" * 60)
    save_sampled_json_files(args.root, sampled_pairs, args.reverse)
    
    # Initialize feature extractor
    extractor = DINOFeatureExtractor('dinov2_vitb14_reg', args.patch_size, device)
    
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
    
    for (take_id, cam, idx), roles in tqdm(images.items(), desc="Extracting features"):
        vid_idx = int(idx)
        
        # Build image path
        img_path = dataset_dir / take_id / cam / f"{vid_idx}.jpg"
        
        if not img_path.exists():
            if args.verbose:
                print(f"WARNING: Image not found: {img_path}")
            errors += 1
            continue
        
        # Create output directory
        out_dir = features_dir / take_id / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract source features if needed
        if roles['is_source']:
            out_path_source = out_dir / f"{vid_idx}_source.npz"
            
            if out_path_source.exists() and not args.overwrite:
                skipped += 1
            else:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))[..., ::-1].copy()
                    target_h, target_w = get_target_size(args.patch_size, args.reverse, is_source=True)
                    img_resized = resize_image(img, target_h, target_w)
                    img_tensor = extractor.transform(img_resized).unsqueeze(0)
                    
                    # Extract features
                    features = extractor.extract(img_tensor)
                    
                    # Save as compressed numpy
                    np.savez_compressed(str(out_path_source), features=features.numpy())
                    processed += 1
                except Exception as e:
                    if args.verbose:
                        print(f"ERROR processing {img_path} (source): {e}")
                    errors += 1
        
        # Extract dest features if needed
        if roles['is_dest']:
            out_path_dest = out_dir / f"{vid_idx}_dest.npz"
            
            if out_path_dest.exists() and not args.overwrite:
                skipped += 1
            else:
                try:
                    # Load and preprocess image (only if not already loaded)
                    if not roles['is_source'] or not out_path_source.exists():
                        img = cv2.imread(str(img_path))[..., ::-1].copy()
                    
                    target_h, target_w = get_target_size(args.patch_size, args.reverse, is_source=False)
                    img_resized = resize_image(img, target_h, target_w)
                    img_tensor = extractor.transform(img_resized).unsqueeze(0)
                    
                    # Extract features
                    features = extractor.extract(img_tensor)
                    
                    # Save as compressed numpy
                    np.savez_compressed(str(out_path_dest), features=features.numpy())
                    processed += 1
                except Exception as e:
                    if args.verbose:
                        print(f"ERROR processing {img_path} (dest): {e}")
                    errors += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output directory: {features_dir}")
    
    # Estimate storage size
    total_size = sum(f.stat().st_size for f in features_dir.rglob("*.npz"))
    print(f"Total storage used: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-extract DINOv2 features for O-MaMa dataset")
    parser.add_argument("--root", type=str, required=True,
                       help="Path to the dataset root directory")
    parser.add_argument("--patch_size", type=int, default=16,
                       help="Patch size of the DINO transformer (default: 16)")
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
    
    args = parser.parse_args()
    precompute_features(args)