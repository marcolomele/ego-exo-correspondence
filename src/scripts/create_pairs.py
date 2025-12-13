import json
import argparse
import random


def sample_frames_from_pairs(pairs, sample_ratio=1.0, target_pairs=None, seed=42):
    """
    Randomly sample frames from pairs to reduce dataset size while maintaining diversity.
    Keeps pairs where both frames are sampled.
    
    Args:
        pairs: List of (aria_rgb, aria_mask, cam_rgb, cam_mask) tuples
        sample_ratio: Fraction of frames to keep (0.0 to 1.0). Ignored if target_pairs is set.
        target_pairs: Exact number of pairs to sample. If set, iteratively finds the right ratio.
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled pairs
    """
    if sample_ratio >= 1.0 and target_pairs is None:
        return pairs
    
    if target_pairs is not None and target_pairs >= len(pairs):
        return pairs
    
    random.seed(seed)
    
    # Collect all unique frames across all pairs
    unique_frames = set()
    for aria_rgb, aria_mask, cam_rgb, cam_mask in pairs:
        # Parse paths: root//take//cam//obj//rgb//idx
        aria_parts = aria_rgb.split('//')
        cam_parts = cam_rgb.split('//')
        
        # Extract frame identifiers: (take_id, camera, object, frame_idx)
        aria_frame = (aria_parts[1], aria_parts[2], aria_parts[3], aria_parts[5])
        cam_frame = (cam_parts[1], cam_parts[2], cam_parts[3], cam_parts[5])
        
        unique_frames.add(aria_frame)
        unique_frames.add(cam_frame)
    
    total_frames = len(unique_frames)
    unique_frames_list = list(unique_frames)
    
    # If target_pairs specified, iteratively find the right frame ratio
    if target_pairs is not None:
        frame_ratio = min(1.0, (target_pairs / len(pairs)) * 1.5)  # Start with estimate
        
        for iteration in range(10):  # Max 10 iterations
            num_samples = int(total_frames * frame_ratio)
            sampled_frames = set(random.sample(unique_frames_list, num_samples))
            
            # Count resulting pairs
            sampled_pairs = [
                pair for pair in pairs
                if (pair[0].split('//')[1], pair[0].split('//')[2], pair[0].split('//')[3], pair[0].split('//')[5]) in sampled_frames
                and (pair[2].split('//')[1], pair[2].split('//')[2], pair[2].split('//')[3], pair[2].split('//')[5]) in sampled_frames
            ]
            
            num_sampled = len(sampled_pairs)
            
            # Within 5% of target or last iteration
            if abs(num_sampled - target_pairs) <= target_pairs * 0.05 or iteration == 9:
                return sampled_pairs
            
            # Adjust ratio
            frame_ratio *= (target_pairs / num_sampled) ** 0.7  # Dampened adjustment
            frame_ratio = min(1.0, frame_ratio)
        
        return sampled_pairs
    
    # Standard ratio-based sampling
    num_samples = int(total_frames * sample_ratio)
    sampled_frames = set(random.sample(unique_frames_list, num_samples))
    
    # Keep pairs where BOTH frames are in the sampled set
    sampled_pairs = []
    for aria_rgb, aria_mask, cam_rgb, cam_mask in pairs:
        aria_parts = aria_rgb.split('//')
        cam_parts = cam_rgb.split('//')
        
        aria_frame = (aria_parts[1], aria_parts[2], aria_parts[3], aria_parts[5])
        cam_frame = (cam_parts[1], cam_parts[2], cam_parts[3], cam_parts[5])
        
        if aria_frame in sampled_frames and cam_frame in sampled_frames:
            sampled_pairs.append((aria_rgb, aria_mask, cam_rgb, cam_mask))
    
    return sampled_pairs


def make_pairs(data_dir, split='train', setting='exoego', sample_ratio=1.0, target_pairs=None, seed=42):
    """
    Create ego-exo frame pairs for a given split with optional sampling.
    
    Args:
        data_dir: Root directory containing split.json and take annotations
        split: Split name ('train', 'val', or 'test')
        setting: Direction ('egoexo' or 'exoego')
        sample_ratio: Fraction of frames to sample (default: 1.0 = no sampling)
        target_pairs: Exact number of pairs to sample (overrides sample_ratio)
        seed: Random seed for reproducibility
    """
    with open(f'{data_dir}/split.json', 'r') as fp:
        splits = json.load(fp)

    split_takes = splits[split]

    # Create all pairs for this split
    pairs = []
    for take in split_takes:
        with open(f'{data_dir}/{take}/annotation.json', 'r') as fp:
            annotation = json.load(fp) 

        for obj_name in annotation['masks']:
            # Find ego camera
            ego_cam = None
            for cam in annotation['masks'][obj_name]:
                if 'aria' in cam:
                    ego_cam = cam
                    break

            if ego_cam is None:
                continue

            # Create pairs with exo cameras
            for cam in annotation['masks'][obj_name]:
                if 'aria' in cam:
                    continue

                # Determine destination camera based on setting
                dcam = ego_cam if setting == 'egoexo' else cam

                for idx in annotation['masks'][obj_name][dcam]:
                    aria_rgb_path = f'{data_dir}//{take}//{ego_cam}//{obj_name}//rgb//{idx}'
                    aria_mask_path = f'{data_dir}//{take}//{ego_cam}//{obj_name}//mask//{idx}'
                    cam_rgb_path = f'{data_dir}//{take}//{cam}//{obj_name}//rgb//{idx}'
                    cam_mask_path = f'{data_dir}//{take}//{cam}//{obj_name}//mask//{idx}'

                    pairs.append((aria_rgb_path, aria_mask_path, cam_rgb_path, cam_mask_path))

    print(f'  Created {len(pairs)} pairs')
    
    # Apply frame-level sampling if requested
    if target_pairs is not None or sample_ratio < 1.0:
        pairs = sample_frames_from_pairs(pairs, sample_ratio=sample_ratio, target_pairs=target_pairs, seed=seed)
        if target_pairs is not None:
            print(f'  Sampled to {len(pairs)} pairs (target: {target_pairs})')
        else:
            print(f'  Sampled to {len(pairs)} pairs ({sample_ratio*100:.1f}% of frames)')
    
    # Save pairs
    output_path = f'{data_dir}/{split}_{setting}_pairs.json'
    with open(output_path, 'w') as fp:
        json.dump(pairs, fp)
    
    return len(pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create ego-exo frame pairs with optional sampling. '
                    'This is the single source of truth for frame sampling.'
    )
    parser.add_argument('--scenario', type=str, required=True, 
                        choices=['cooking', 'health'],
                        help='Scenario to process (cooking or health)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Fraction of frames to sample for all splits (0.0 to 1.0, default: 1.0 = no sampling)')
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='Fraction of frames to sample for train split (overrides --sample_ratio)')
    parser.add_argument('--val_ratio', type=float, default=None,
                        help='Fraction of frames to sample for val split (overrides --sample_ratio)')
    parser.add_argument('--test_ratio', type=float, default=None,
                        help='Fraction of frames to sample for test split (overrides --sample_ratio)')
    parser.add_argument('--train_pairs', type=int, default=None,
                        help='Exact number of pairs for train split (overrides --train_ratio)')
    parser.add_argument('--val_pairs', type=int, default=None,
                        help='Exact number of pairs for val split (overrides --val_ratio)')
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Exact number of pairs for test split (overrides --test_ratio)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    data_dir = f'output_dir_{args.scenario}'
    
    # Load available splits
    with open(f'{data_dir}/split.json', 'r') as fp:
        available_splits = json.load(fp)
    
    # Determine sampling parameters for each split (target pairs override ratios)
    split_params = {}
    for split_name in ['train', 'val', 'test']:
        target_pairs = getattr(args, f'{split_name}_pairs', None)
        ratio = getattr(args, f'{split_name}_ratio', None)
        
        if target_pairs is not None:
            split_params[split_name] = {'target_pairs': target_pairs, 'ratio': 1.0}
        elif ratio is not None:
            split_params[split_name] = {'target_pairs': None, 'ratio': ratio}
        else:
            split_params[split_name] = {'target_pairs': None, 'ratio': args.sample_ratio}
    
    print(f"Creating pairs for scenario: {args.scenario}")
    print(f"Available splits: {list(available_splits.keys())}")
    print(f"Total takes: {sum(len(v) for v in available_splits.values())}")
    
    # Display sampling parameters
    for split_name in ['train', 'val', 'test']:
        params = split_params[split_name]
        if params['target_pairs'] is not None:
            print(f"  {split_name}: target {params['target_pairs']} pairs")
        else:
            print(f"  {split_name}: {params['ratio']*100:.0f}% of frames")
    
    print(f"Random seed: {args.seed}")
    print()
    
    # Process all splits with exoego setting
    total_pairs = 0
    for split in available_splits.keys():
        params = split_params.get(split, {'target_pairs': None, 'ratio': args.sample_ratio})
        
        print(f"Processing {split} split:")
        num_pairs = make_pairs(
            data_dir, split, 'exoego',
            sample_ratio=params['ratio'],
            target_pairs=params['target_pairs'],
            seed=args.seed
        )
        total_pairs += num_pairs
        print()
    
    print(f"Total pairs created: {total_pairs}")
