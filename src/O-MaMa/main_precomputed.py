"""
Training script for O-MaMa using precomputed DINOV2, DINOv3 or ResNet-50 features.

This is a modified version of main.py that:
- Uses precomputed features loaded from disk (10x faster training)
- Uses DescriptorExtractorPrecomputed (no DINO model)
- Uses Masks_Dataset_Precomputed (loads .npz features)
- Increases num_workers to 4 (since I/O is lighter)
- Adds feature projection parameters to optimizer

Prerequisites:
    Run precompute_features_{dinov2, dinov3, resnet50}.py first to extract DINOV2, DINOv3 or ResNet-50 features.

Usage for DINOV2 features:
    python main_precomputed.py --root ../../data/root --reverse --patch_size 14 --dino_feat_dim 768
    
Usage for DINOv3 features:
    python main_precomputed.py --root ../../data/root --reverse --patch_size 16

Usage for ResNet-50 features:
    python main_precomputed.py --root ../../data/root --reverse  --dino_feat_dim 2048
"""


import torch
import argparse
from descriptors.get_descriptors_precomputed import DescriptorExtractorPrecomputed
from dataset.dataset_masks_precomputed import Masks_Dataset_Precomputed
from model.model import Attention_projector
from evaluation.evaluate import add_to_json, evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import helpers
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime
import logging
import numpy as np


def save_checkpoint(model, path, description="checkpoint", additional_state=None, separate_projection=True):
    """
    Safely save model checkpoint with error handling and verification.
    
    Args:
        model: PyTorch model to save
        path: Full path where to save the checkpoint
        description: Human-readable description for logging
        additional_state: Optional dictionary with additional state to save
        separate_projection: If True and feature_proj_state_dict exists, save it separately
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Check if we need to save projection separately
        proj_state_dict = None
        if separate_projection and additional_state and 'feature_proj_state_dict' in additional_state:
            proj_state_dict = additional_state.pop('feature_proj_state_dict')
        
        state_dict = {
            'model_state_dict': model.state_dict(),
        }
        if additional_state:
            state_dict.update(additional_state)
        
        # Save model checkpoint
        torch.save(state_dict, path)
        
        # Save projection separately if needed
        if proj_state_dict is not None:
            # Replace 'model_weights' with 'projection_weights' in path
            proj_path = path.replace('model_weights', 'projection_weights')
            os.makedirs(os.path.dirname(proj_path), exist_ok=True)
            torch.save({'feature_proj_state_dict': proj_state_dict}, proj_path)
            logging.info(f"Saved projection weights to {proj_path}")
        
        if not os.path.exists(path):
            logging.error(f"Failed to save {description}: File does not exist after save attempt: {path}")
            return False
        
        file_size = os.path.getsize(path)
        if file_size == 0:
            logging.error(f"Failed to save {description}: File is empty: {path}")
            return False
        
        # Verify projection checkpoint if it was saved separately
        if proj_state_dict is not None:
            proj_path = path.replace('model_weights', 'projection_weights')
            if not os.path.exists(proj_path):
                logging.error(f"Failed to save projection weights: File does not exist: {proj_path}")
                return False
            proj_size = os.path.getsize(proj_path)
            if proj_size == 0:
                logging.error(f"Failed to save projection weights: File is empty: {proj_path}")
                return False
            logging.info(f"Successfully saved projection to {proj_path} (size: {proj_size / 1024:.2f} KB)")
        
        logging.info(f"Successfully saved {description} to {path} (size: {file_size / (1024*1024):.2f} MB)")
        return True
    
    except Exception as e:
        logging.error(f"Error saving {description} to {path}: {str(e)}")
        return False


def save_json(data, path, description="JSON"):
    """
    Safely save JSON data with error handling and verification.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        
        if not os.path.exists(path):
            logging.error(f"Failed to save {description}: File does not exist after save attempt: {path}")
            return False
        
        file_size = os.path.getsize(path)
        if file_size == 0:
            logging.error(f"Failed to save {description}: File is empty: {path}")
            return False
        
        logging.info(f"Successfully saved {description} to {path} (size: {file_size / 1024:.2f} KB)")
        return True
    
    except Exception as e:
        logging.error(f"Error saving {description} to {path}: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train O-MaMa with precomputed features")
    parser.add_argument("--root", type=str, default="../../data/root", help="Path to the dataset")
    parser.add_argument("--features_dir", type=str, default="../../data/root/precomputed_features", help="Path to precomputed features")
    parser.add_argument("--output_dir", type=str, default="train_output", help="Output directory")
    parser.add_argument("--dino_feat_dim", type=int, default=768, help="Extractor model feature dimension (default: 768 for DINOv2)")
    parser.add_argument("--target_feat_dim", type=int, default=768, help="Target feature dimension (default: 768)")
    
    parser.add_argument("--reverse", action="store_true", help="Flag to select exo->ego pairs")
    
    parser.add_argument("--order", default=2, type=int, help="order of adjacency matrix, 2 for 2nd order")
    parser.add_argument("--devices", default="0", type=str)
    parser.add_argument("--exp_name", type=str, default="Train_OMAMA_Precomputed")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size of the dino transformer")
    parser.add_argument("--context_size", type=int, default=20, help="Size of the context for the object")
    parser.add_argument("--N_masks_per_batch", default=32, type=int)
    
    parser.add_argument("--N_epochs", default=10, type=int)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers (default: 4)")
    args = parser.parse_args()

    # Setup logging to both console and file
    now = datetime.now()
    run_folder = f"run_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
    
    if args.output_dir is not None:
        base_output_dir = Path(args.output_dir).resolve()
    else:
        script_dir = Path(__file__).parent.resolve()
        base_output_dir = script_dir / "train_output"
    
    output_dir = base_output_dir / run_folder
    folder_weights = output_dir / "model_weights"
    
    try:
        folder_weights.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        print(f"Model weights will be saved to: {folder_weights}")
    except Exception as e:
        print(f"ERROR: Failed to create output directories: {e}")
        print(f"Attempted path: {output_dir}")
        sys.exit(1)
    
    log_file = output_dir / f"training_{run_folder}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Starting training run: {run_folder}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Command line arguments: {vars(args)}")
    logging.info("=" * 60)
    logging.info("USING PRECOMPUTED FEATURES PIPELINE")
    logging.info("=" * 60)
    
    # Verify output directory is writable
    test_file = output_dir / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        logging.info("Output directory is writable")
    except Exception as e:
        logging.error(f"Output directory is NOT writable: {e}")
        sys.exit(1)

    helpers.set_all_seeds(42)
    if args.devices != "cpu":
        gpus = [args.devices]
        device_ids = [f'cuda:{gpu}' for gpu in gpus]
        device = torch.device(f'cuda:{device_ids[0].split(":")[1]}') if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    logging.info(f"Using device: {device}")
    
    # Determine features directory
    features_dir = args.features_dir if args.features_dir else os.path.join(args.root, 'precomputed_features')
    logging.info(f"Loading precomputed features from: {features_dir}")
    
    # Training dataset with precomputed features
    logging.info("Loading training dataset (precomputed features)...")
    train_dataset = Masks_Dataset_Precomputed(
        args.root, args.patch_size, args.reverse, 
        N_masks_per_batch=args.N_masks_per_batch, 
        order=args.order, train=True, test=False,
        features_dir=features_dir
    )
    percent_train = max(1, int(1 * len(train_dataset)))   
    from torch.utils.data import Subset
    train_dataset_subset = Subset(train_dataset, list(range(percent_train)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_subset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=helpers.our_collate_fn, 
        num_workers=args.num_workers,  # Increased from 1 to 4
        pin_memory=True
    )
    logging.info(f"Training dataset loaded: {len(train_dataset_subset)} samples ({percent_train}), {len(train_dataloader)} batches")
    logging.info(f"Using {args.num_workers} data loading workers")
    
    # Validation dataset with precomputed features
    logging.info("Loading validation dataset (precomputed features)...")
    val_dataset = Masks_Dataset_Precomputed(
        args.root, args.patch_size, args.reverse, 
        N_masks_per_batch=args.N_masks_per_batch, 
        order=args.order, train=False, test=False,
        features_dir=features_dir
    )
    percent_val = max(1, int(1 * len(val_dataset)))
    val_dataset_subset = Subset(val_dataset, list(range(percent_val)))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_subset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=helpers.our_collate_fn
    )
    logging.info(f"Validation dataset loaded: {len(val_dataset_subset)} samples ({percent_val})")

    best_IoU = 0

    logging.info("Initializing model and descriptor extractor...")
    
    # Use precomputed descriptor extractor (no DINO model!)
    descriptor_extractor = DescriptorExtractorPrecomputed(
        args.patch_size, args.context_size, device,
        dino_feat_dim=args.dino_feat_dim,
        target_feat_dim=args.target_feat_dim
    )
    logging.info(f"Descriptor extractor initialized (projection: {args.dino_feat_dim} -> {args.target_feat_dim})")
    
    model = Attention_projector(reverse=args.reverse).to(device)
    logging.info(f"Model:\n{model}")

    # Collect all trainable parameters (model + feature projection)
    trainable_params = list(model.parameters())
    if descriptor_extractor.feature_proj is not None:
        trainable_params.extend(descriptor_extractor.feature_proj.parameters())
        logging.info(f"Added feature projection layer to optimizer ({args.dino_feat_dim} -> {args.target_feat_dim})")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=8e-6)
    T_max = args.N_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=1e-6)

    train_losses = []
    val_losses = []
    val_metrics_history = []

    for epoch in range(args.N_epochs):
        logging.info(f'===== Starting epoch {epoch+1}/{args.N_epochs} - Training =====')
        epoch_train_losses = []
        model.train()
        
        # Set projection layer to train mode
        if descriptor_extractor.feature_proj is not None:
            descriptor_extractor.feature_proj.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")):
            if batch is None:
                continue
            
            # Debug: Check for NaN/Inf in input features
            if batch_idx == 0 and epoch == 0:
                source_feat = batch.get('SOURCE_features', batch.get('source_feat'))
                dest_feat = batch.get('DEST_features', batch.get('dest_feat'))
                logging.info(f"Source features shape: {source_feat.shape}, range: [{source_feat.min():.3f}, {source_feat.max():.3f}]")
                logging.info(f"Dest features shape: {dest_feat.shape}, range: [{dest_feat.min():.3f}, {dest_feat.max():.3f}]")
                if torch.isnan(source_feat).any() or torch.isnan(dest_feat).any():
                    logging.error("NaN detected in input features!")
                logging.info(f"POS_mask_position: {batch['POS_mask_position']}, is_visible: {batch['is_visible']}")
                logging.info(f"DEST_SAM_masks shape: {batch['DEST_SAM_masks'].shape}")
            
            DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
            SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
            
            # Debug descriptors on first batch
            if batch_idx == 0 and epoch == 0:
                logging.info(f"SOURCE_descriptors shape: {SOURCE_descriptors.shape}, range: [{SOURCE_descriptors.min():.3f}, {SOURCE_descriptors.max():.3f}]")
                logging.info(f"DEST_descriptors shape: {DEST_descriptors.shape}, range: [{DEST_descriptors.min():.3f}, {DEST_descriptors.max():.3f}]")
                logging.info(f"SOURCE_img_feats shape: {SOURCE_img_feats.shape}")
                logging.info(f"DEST_img_feats shape: {DEST_img_feats.shape}")
                if torch.isnan(SOURCE_descriptors).any():
                    logging.error("NaN in SOURCE_descriptors!")
                if torch.isnan(DEST_descriptors).any():
                    logging.error("NaN in DEST_descriptors!")
            
            best_similarities, best_masks, refined_mask, loss, top5_masks = model(
                SOURCE_descriptors, DEST_descriptors, 
                SOURCE_img_feats, DEST_img_feats, 
                batch['POS_mask_position'], batch['is_visible'],
                batch['DEST_SAM_masks'], test_mode=False
            )
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_train_losses.append(loss.item())

        epoch_train_loss_mean = float(sum(epoch_train_losses) / (len(epoch_train_losses) if len(epoch_train_losses) > 0 else 1))
        train_losses.append(epoch_train_loss_mean)
        logging.info(f'Epoch {epoch+1} training loss: {epoch_train_loss_mean:.6f}')
        
        # Save last epoch checkpoint with projection layer state
        last_checkpoint_path = os.path.join(folder_weights, f'last_epoch_{run_folder}.pt')
        additional_state = {}
        if descriptor_extractor.feature_proj is not None:
            additional_state['feature_proj_state_dict'] = descriptor_extractor.feature_proj.state_dict()
        save_checkpoint(model, last_checkpoint_path, f"last epoch checkpoint (epoch {epoch+1})", additional_state)
        
        logging.info(f'===== Starting epoch {epoch+1}/{args.N_epochs} - Validation =====')
        processed_epoch, pred_json_epoch, gt_json_epoch = {}, {}, {}
        epoch_val_losses = []
        model.eval()
        
        # Set projection layer to eval mode
        if descriptor_extractor.feature_proj is not None:
            descriptor_extractor.feature_proj.eval()
        
        for idx, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation")):
            with torch.no_grad():
                if batch is None:
                    continue
                    
                DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
                SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
                
                similarities, pred_masks_idx, refined_mask, loss, top5_masks = model(
                    SOURCE_descriptors, DEST_descriptors, 
                    SOURCE_img_feats, DEST_img_feats, 
                    batch['POS_mask_position'], batch['is_visible'],
                    batch['DEST_SAM_masks'], test_mode=False
                )
                
                pred_mask = refined_mask.squeeze().detach().cpu().numpy()
                confidence = similarities.detach().cpu().numpy()
                
                epoch_val_losses.append(loss.item())
                pred_json_epoch, gt_json_epoch = add_to_json(
                    val_dataset, batch['pair_idx'], 
                    pred_mask, confidence,
                    processed_epoch, pred_json_epoch, gt_json_epoch
                )

        epoch_val_loss_mean = float(sum(epoch_val_losses) / (len(epoch_val_losses) if len(epoch_val_losses) > 0 else 1))
        val_losses.append(epoch_val_loss_mean)
        logging.info(f'Epoch {epoch+1} validation loss: {epoch_val_loss_mean:.6f}')
        
        logging.info(f'Computing epoch {epoch+1} validation metrics...')
        aggregated_metrics, per_observation_metrics = evaluate(gt_json_epoch, pred_json_epoch, args.reverse)
        val_metrics_history.append(aggregated_metrics)
        
        logging.info(f"Epoch {epoch+1} validation metrics (aggregated):")
        for metric_name, metric_value in aggregated_metrics.items():
            logging.info(f"  {metric_name}: {metric_value:.6f}")
        
        logging.info(f"Epoch {epoch+1} per-observation statistics:")
        logging.info(f"  Total observations: {len(per_observation_metrics['iou_per_obs'])}")
        if len(per_observation_metrics['iou_per_obs']) > 0:
            iou_std = float(np.std(per_observation_metrics['iou_per_obs']))
            logging.info(f"  IoU std: {iou_std:.6f}")
        
        # Save best model checkpoint
        if aggregated_metrics['iou'] > best_IoU:
            best_IoU = aggregated_metrics['iou']
            best_checkpoint_path = os.path.join(folder_weights, f'best_IoU_{run_folder}.pt')
            additional_state = {}
            if descriptor_extractor.feature_proj is not None:
                additional_state['feature_proj_state_dict'] = descriptor_extractor.feature_proj.state_dict()
            save_checkpoint(model, best_checkpoint_path, f"best IoU checkpoint (epoch {epoch+1}, IoU={best_IoU:.6f})", additional_state)
        
        # Save epoch validation results
        epoch_results_path = os.path.join(output_dir, f'val_results_epoch{epoch+1}.json')
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss_mean,
            "val_loss": epoch_val_loss_mean,
            "aggregated_metrics": aggregated_metrics,
            "per_observation_metrics": per_observation_metrics
        }
        save_json(epoch_results, epoch_results_path, f"epoch {epoch+1} validation results")

    # Save training and validation history
    logging.info("Saving final training statistics...")
    training_stats = {
        "exp_name": args.exp_name,
        "run_folder": run_folder,
        "args": vars(args),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics": val_metrics_history,
        "best_iou": float(best_IoU),
        "pipeline": "precomputed_features"
    }
    stats_save_path = os.path.join(output_dir, f'training_stats_{run_folder}.json')
    save_json(training_stats, stats_save_path, "final training statistics")
    
    # Final summary
    logging.info("=" * 80)
    logging.info("Training completed successfully!")
    logging.info(f"Pipeline: PRECOMPUTED FEATURES")
    logging.info(f"Best validation IoU: {best_IoU:.6f}")
    logging.info(f"All results saved to: {output_dir}")
    logging.info(f"Model weights saved to: {folder_weights}")
    logging.info("=" * 80)

