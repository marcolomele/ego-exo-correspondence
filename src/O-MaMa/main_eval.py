""" Defines the main script for evaluating O-MaMa """

import torch
import argparse
import numpy as np
import json
from descriptors.get_descriptors import DescriptorExtractor
from dataset.dataset_masks import Masks_Dataset
from model.model import Attention_projector
from evaluation.evaluate import add_to_json, evaluate
from pathlib import Path

import helpers
from datetime import datetime
from tqdm import tqdm
import os
import sys
import logging

def save_json(data, path, description="JSON"):
    """
    Safely save JSON data with error handling and verification.
    
    Args:
        data: Data to save as JSON
        path: Full path where to save the JSON
        description: Human-readable description for logging
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save JSON with explicit flush
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Verify file was written and has non-zero size
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


def compute_IoU(pred_mask, gt_mask):
    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    union = torch.logical_or(pred_mask, gt_mask).sum()
    IoU = intersection / (union + 1e-6)
    return IoU

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match masks from ego-exo pairs")
    parser.add_argument("--root", type=str, default="../../data/root",help="Path to the dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="Exo2Ego_weights_checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="Output directory")

    parser.add_argument("--reverse", action="store_true", help="Flag to select exo->ego pairs")
    
    parser.add_argument("--order", default=2, type=int, help="order of adjacency matrix, 2 for 2nd order")
    parser.add_argument("--devices", default="0", type=str)
    parser.add_argument("--exp_name", type=str, default="Evaluation_OMAMA_Ego->Exo")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size of the dino transformer")
    parser.add_argument("--context_size", type=int, default=20, help="Size of the context sizo for the object")
    parser.add_argument("--test_on_subset", type=int, default=None, help="Limit the test dataset to the first N items")
    parser.add_argument("--N_masks_per_batch", default=32, type=int)
    args = parser.parse_args()

    # Setup logging to both console and file
    now = datetime.now()
    run_folder = f"run_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
    
    # Use absolute path for output directory
    if args.output_dir is not None:
        base_output_dir = Path(args.output_dir).resolve()
    else:
        # Get absolute path of script directory
        script_dir = Path(__file__).parent.resolve()
        base_output_dir = script_dir / "eval_output"
    
    output_dir = base_output_dir / run_folder
    
    # Create directories with error handling
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"ERROR: Failed to create output directories: {e}")
        print(f"Attempted path: {output_dir}")
        sys.exit(1)
    
    # Setup logging
    log_file = output_dir / f"evaluation_{run_folder}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Starting evaluation run: {run_folder}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Command line arguments: {vars(args)}")
    
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
        gpus = [args.devices]  # Specify which GPUs to use
        device_ids = [f'cuda:{gpu}' for gpu in gpus]
        device = torch.device(f'cuda:{device_ids[0].split(":")[1]}') if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    logging.info(f"Using device: {device}")

    # The test dataset is the full validation dataset, used for the final evaluation
    logging.info("Loading test dataset...")
    test_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, train=False, N_masks_per_batch=args.N_masks_per_batch, order=args.order, test=True)
    
    # Optionally limit to first N frames/items by slicing the dataset, if --test_on_subset is provided
    if args.test_on_subset is not None:
        limited_indices = list(range(min(args.test_on_subset, len(test_dataset))))
        from torch.utils.data import Subset
        test_dataset_limited = Subset(test_dataset, limited_indices)
        logging.info(f"Limiting test dataset to first {args.test_on_subset} items for evaluation.")
        test_dataloader = torch.utils.data.DataLoader(test_dataset_limited, batch_size=1, shuffle=False, collate_fn=helpers.our_collate_fn, num_workers=1, pin_memory=True)
        logging.info(f"Test dataset loaded: {len(test_dataset_limited)} samples (limited from {len(test_dataset)})")
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=helpers.our_collate_fn, num_workers=1, pin_memory=True)
        logging.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    logging.info("Initializing descriptor extractor and model...")
    descriptor_extractor = DescriptorExtractor('dinov2_vitb14_reg', args.patch_size, args.context_size, device)
    model = Attention_projector(args.reverse).to(device)
    logging.info(f"Model:\n{model}")

    logging.info(f"Loading checkpoint from: {args.checkpoint_dir}")
    try:
        checkpoint_weights = torch.load(args.checkpoint_dir, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_weights, strict=False)
        logging.info(f"Successfully loaded checkpoint from {args.checkpoint_dir}")
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {args.checkpoint_dir}: {str(e)}")
        sys.exit(1)
    
    logging.info("Starting evaluation...")
    processed_test, pred_json_test, gt_json_test = {}, {}, {}
    test_losses = []
    
    model.eval()
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluation")):
        with torch.no_grad():
            # If our_collate_fn returns None because all batch samples are None, skip this batch
            if batch is None:
                continue
            
            DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
            SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
            
            is_visible_GT = batch['is_visible']
            POS_mask_position_GT = batch['POS_mask_position']
            similarities, pred_masks_idx, pred_mask, loss, top5_masks = model(SOURCE_descriptors, DEST_descriptors, 
                                                                                                    SOURCE_img_feats, DEST_img_feats, 
                                                                                                    batch['POS_mask_position'], batch['is_visible'],
                                                                                                    batch['DEST_SAM_masks'], test_mode = True)
            
            pred_mask = pred_mask.squeeze().detach().cpu().numpy()
            confidence = similarities.detach().cpu().numpy()
            
            if loss is not None:
                test_losses.append(loss.item())
            
            pred_json_test, gt_json_test = add_to_json(test_dataset, batch['pair_idx'], 
                                                       pred_mask, confidence,
                                                       processed_test, pred_json_test, gt_json_test)
    
    # Calculate average test loss if available
    if len(test_losses) > 0:
        test_loss_mean = float(sum(test_losses) / len(test_losses))
        logging.info(f"Average test loss: {test_loss_mean:.6f}")
    else:
        logging.info("No test loss values collected")

    logging.info("Computing evaluation metrics...")
    final_json_gt = {"version": "xx",
                    "challenge": "xx",
                    "annotations": gt_json_test}

    aggregated_metrics, per_observation_metrics = evaluate(gt_json_test, pred_json_test, args.reverse)
    
    # Log all aggregated metrics
    logging.info("Evaluation metrics (aggregated):")
    for metric_name, metric_value in aggregated_metrics.items():
        logging.info(f"  {metric_name}: {metric_value:.6f}")
    
    # Log per-observation statistics
    logging.info("Per-observation statistics:")
    logging.info(f"  Total observations: {len(per_observation_metrics['iou_per_obs'])}")
    if len(per_observation_metrics['iou_per_obs']) > 0:
        iou_std = float(np.std(per_observation_metrics['iou_per_obs']))
        logging.info(f"  IoU std: {iou_std:.6f}")

    # Save metrics with both aggregated and per-observation metrics
    metrics_save_path = os.path.join(output_dir, f'results_metrics_{run_folder}.json')
    metrics_results = {
        "exp_name": args.exp_name,
        "run_folder": run_folder,
        "args": vars(args),
        "checkpoint_path": args.checkpoint_dir,
        "test_loss": test_loss_mean if len(test_losses) > 0 else None,
        "aggregated_metrics": aggregated_metrics,
        "per_observation_metrics": per_observation_metrics
    }
    save_json(metrics_results, metrics_save_path, "evaluation metrics")

    # Saving the json with the results
    logging.info("Saving prediction and ground truth JSON files...")
    if args.reverse:
        final_json = {'exo-ego':{'results': pred_json_test}}
        assert "exo-ego" in final_json
        preds = final_json["exo-ego"]

        assert type(preds) == type({})
        for key in ["results"]:
            assert key in preds.keys()

        save_path = os.path.join(output_dir, 'exo2ego_predictions_' + args.exp_name + '.json')
        save_path_gt = os.path.join(output_dir, 'exo2egoGT.json')
    else:
        final_json = {'ego-exo':{'results': pred_json_test}}
        assert "ego-exo" in final_json
        preds = final_json["ego-exo"]

        assert type(preds) == type({})
        for key in ["results"]:
            assert key in preds.keys()
        
        save_path = os.path.join(output_dir, 'ego2exo_predictions_' + args.exp_name + '.json')
        save_path_gt = os.path.join(output_dir, 'ego2exoGT.json')

    save_json(convert_ndarray(final_json), save_path, "predictions JSON")
    save_json(convert_ndarray(final_json_gt), save_path_gt, "ground truth JSON")
    
    # Final summary
    logging.info("=" * 80)
    logging.info("Evaluation completed successfully!")
    if 'iou' in aggregated_metrics:
        logging.info(f"Best IoU: {aggregated_metrics['iou']:.6f}")
    else:
        logging.info("Best IoU: N/A")
    logging.info(f"All results saved to: {output_dir}")
    logging.info(f"Metrics saved to: {metrics_save_path}")
    logging.info(f"Predictions saved to: {save_path}")
    logging.info(f"Ground truth saved to: {save_path_gt}")
    logging.info("=" * 80)
    