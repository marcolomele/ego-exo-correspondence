"""
Dataloader for the Ego-Exo4D correspondences dataset with precomputed DINOv3 features.

This is a modified version of dataset_masks.py that:
- Loads precomputed features from .npz files instead of images
- Skips image loading entirely for maximum speed
- Maintains the same mask/bbox loading logic
"""

import os 
import torch
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
import torch.nn.functional as F

from torch.utils.data import Dataset
import random

from dataset.adj_descriptors import get_adj_matrix
from dataset.dataset_utils import compute_IoU, compute_IoU_bbox, bbox_from_mask


class Masks_Dataset_Precomputed(Dataset):
    """
    Dataset that loads precomputed DINOv3 features instead of images.
    
    This provides 3-5x faster training by eliminating the DINO forward pass bottleneck.
    Features must be pre-extracted using precompute_features.py before using this dataset.
    """
    
    def __init__(self, root, patch_size, reverse, N_masks_per_batch, order, train, test, features_dir=None):
        """
        Args:
            root: Path to the dataset root
            patch_size: Patch size of the DINO transformer
            reverse: True for exo->ego, False for ego->exo
            N_masks_per_batch: Number of masks per batch
            order: Order of adjacency matrix (2 for 2nd order)
            train: True for training mode
            test: True for test mode (if False and train=False, uses validation)
            features_dir: Path to precomputed features directory (default: {root}/precomputed_features)
        """
        self.root = root
        self.train_mode = train
        self.test_mode = test
        self.reverse = reverse

        # Select the pre-extracted masks directory based on the train/test mode and reverse flag
        if train:
            if reverse:
                self.masks_dir = os.path.join(root, 'Masks_TRAIN_EXO2EGO')
            else:
                self.masks_dir = os.path.join(root, 'Masks_TRAIN_EGO2EXO')
        else:
            if test:
                if reverse:
                    self.masks_dir = os.path.join(root, 'Masks_TEST_EXO2EGO')
                else:
                    self.masks_dir = os.path.join(root, 'Masks_TEST_EGO2EXO')
            else:
                if reverse:
                    self.masks_dir = os.path.join(root, 'Masks_VAL_EXO2EGO')
                else:   
                    self.masks_dir = os.path.join(root, 'Masks_VAL_EGO2EXO')

        # Preprocessed dataset directory
        self.dataset_dir = os.path.join(root, 'processed')
        
        # Precomputed features directory
        if features_dir is not None:
            self.features_dir = features_dir
        else:
            self.features_dir = os.path.join(root, 'precomputed_features')
        
        if not os.path.exists(self.features_dir):
            raise RuntimeError(
                f"Precomputed features directory not found: {self.features_dir}\n"
                f"Please run precompute_features.py first to extract DINOv3 features."
            )

        # Configs for loading the features
        self.N_masks_per_batch = N_masks_per_batch
        self.patch_size = patch_size
        self.order = order

        # Load the mask annotations and pairs
        self.mask_annotations = self.load_mask_annotations()
        self.pairs = self.load_all_pairs()
        self.takes_json = json.load(open(os.path.join(root, 'takes.json'), 'r')) if os.path.exists(os.path.join(root, 'takes.json')) else None

        # Calculate expected image sizes based on patch_size
        self._compute_expected_sizes()

        print(len(self.takes_json), 'TAKES') if self.takes_json is not None else print('NO TAKE JSON FILE FOUND')
        print(f"Using precomputed features from: {self.features_dir}")

    def _compute_expected_sizes(self):
        """Compute expected image sizes based on patch_size and direction."""
        if self.reverse:
            # Reverse mode: exo->ego
            # Source (exo): 38*68 patches
            self.source_h = 38 * self.patch_size
            self.source_w = 68 * self.patch_size
            # Dest (ego): 50*50 patches
            self.dest_h = 50 * self.patch_size
            self.dest_w = 50 * self.patch_size
        else:
            # Normal mode: ego->exo
            # Source (ego): 50*50 patches
            self.source_h = 50 * self.patch_size
            self.source_w = 50 * self.patch_size
            # Dest (exo): 38*68 patches
            self.dest_h = 38 * self.patch_size
            self.dest_w = 68 * self.patch_size

    def load_all_pairs(self):
        """Load the json with the pairs."""
        if self.train_mode:
            if self.reverse:
                pairs_json = 'train_exoego_pairs.json'
            else:
                pairs_json = 'train_egoexo_pairs.json'
        else:
            if self.test_mode:
                if self.reverse:
                    pairs_json = 'test_exoego_pairs.json'
                else:
                    pairs_json = 'test_egoexo_pairs.json'
            else:    
                if self.reverse:
                    pairs_json = 'val_exoego_pairs.json'
                else:
                    pairs_json = 'val_egoexo_pairs.json'

        print('----------------------------We are loading: ', pairs_json, 'with the pair of images')
        pairs = []
        jsons_dir = os.path.join(self.root, 'dataset_jsons')
        with open(os.path.join(jsons_dir, pairs_json), 'r') as fp:
            pairs.extend(json.load(fp))
        print('LEN OF THE DATASET:', len(pairs))
        return pairs
    
    def load_mask_annotations(self):
        """Load the GT mask annotations."""
        d = self.dataset_dir
        with open(f'{d}/split.json', 'r') as fp:
            splits = json.load(fp)
        valid_takes = splits['train'] + splits['val'] + splits['test']

        annotations = {}
        for take in valid_takes:
            try:
                with open(f'{d}/{take}/annotation.json', 'r') as fp:
                    annotations[take] = json.load(fp)
            except:
                continue
        return annotations

    def select_adjacent_negatives(self, adj_matrix, SAM_bboxes, SAM_masks, mask_GT):
        """Select adjacent negatives based on the adjacency matrix."""
        bbox_GT, _ = bbox_from_mask(mask_GT)
        bbox_iou = compute_IoU_bbox(SAM_bboxes, bbox_GT)
        max_index = torch.argmax(bbox_iou)
        
        # Get the neighbors of the best mask
        adj_matrix[max_index, max_index] = 0
        neighbors = torch.where(adj_matrix[max_index] == 1)[0]
        N_adjacent_indices = self.N_masks_per_batch - 1
        if len(neighbors) > N_adjacent_indices:
            random_indices = np.random.choice(neighbors, N_adjacent_indices, replace=False)
            adjacent_SAM_masks = SAM_masks[random_indices]
            adjacent_SAM_bboxes = SAM_bboxes[random_indices]
        else:
            adjacent_SAM_masks = SAM_masks[neighbors]
            adjacent_SAM_bboxes = SAM_bboxes[neighbors]
            
            # Get remaining negatives
            N_remaining_indices = N_adjacent_indices - len(neighbors)
            if SAM_masks.shape[0] < N_remaining_indices:
                remaining_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=True)
            else:
                remaining_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=False)
            
            adjacent_SAM_masks = torch.cat((adjacent_SAM_masks, SAM_masks[remaining_indices]), dim=0)
            adjacent_SAM_bboxes = torch.cat((adjacent_SAM_bboxes, SAM_bboxes[remaining_indices]), dim=0)
        
        return adjacent_SAM_masks, adjacent_SAM_bboxes

    def get_best_mask(self, SAM_masks, mask_GT):
        """Select the best SAM mask based on IoU with GT."""
        iou = compute_IoU(SAM_masks, mask_GT)
        return torch.argmax(iou)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx_sample):
        # Get the pair of images, 1 refers to the source image and 2 to the destination image
        if self.reverse:
            # img_pth2 ego, img_pth1 exo
            img_pth2, _, img_pth1, _ = self.pairs[idx_sample]
        else:
            # img_pth2 exo, img_pth1 ego
            img_pth1, _, img_pth2, _ = self.pairs[idx_sample]

        root, take_id, cam, obj, _, idx = img_pth1.split('//')
        root = self.dataset_dir
        root2, take_id2, cam2, obj2, _, idx2 = img_pth2.split('//')
        root2 = self.dataset_dir

        # Both viewpoints should have the same take_id, object and index   
        assert obj == obj2
        assert idx == idx2
        assert take_id == take_id2

        vid_idx = int(idx)
        vid_idx2 = int(idx2)

        # ============================================
        # LOAD PRECOMPUTED SOURCE FEATURES
        # ============================================
        source_feat_path = f"{self.features_dir}/{take_id}/{cam}/{vid_idx}_source.npz"
        try:
            source_data = np.load(source_feat_path)
            SOURCE_features = torch.from_numpy(source_data['features']).squeeze(0)  # [C, H, W]
        except FileNotFoundError:
            print(f"WARNING: Source features not found: {source_feat_path}")
            return None
        
        # Set image sizes based on expected dimensions
        self.h1, self.w1 = self.source_h, self.source_w

        # Load the source mask
        mask_annotation_SOURCE = self.mask_annotations[take_id]
        mask_SOURCE = mask_utils.decode(mask_annotation_SOURCE['masks'][obj][cam][idx])
        mask_SOURCE = cv2.resize(mask_SOURCE, (self.w1, self.h1), interpolation=cv2.INTER_NEAREST)
        mask_SOURCE = torch.from_numpy(mask_SOURCE.astype(np.uint8))
        
        # ============================================
        # LOAD PRECOMPUTED DEST FEATURES
        # ============================================
        dest_feat_path = f"{self.features_dir}/{take_id2}/{cam2}/{vid_idx2}_dest.npz"
        try:
            dest_data = np.load(dest_feat_path)
            DEST_features = torch.from_numpy(dest_data['features']).squeeze(0)  # [C, H, W]
        except FileNotFoundError:
            print(f"WARNING: Dest features not found: {dest_feat_path}")
            return None
        
        # Set image sizes based on expected dimensions
        self.h2, self.w2 = self.dest_h, self.dest_w
                
        # Load the destination GT mask
        mask_annotation_DEST = self.mask_annotations[take_id2]
        if idx in mask_annotation_DEST['masks'][obj2][cam2]:
            mask2_GT = mask_utils.decode(mask_annotation_DEST['masks'][obj2][cam2][idx])
            mask2_GT = cv2.resize(mask2_GT, (self.w2, self.h2), interpolation=cv2.INTER_NEAREST)
        else:
            mask2_GT = np.zeros((self.h2, self.w2))
        mask2_GT = torch.from_numpy(mask2_GT.astype(np.uint8))

        # Load the proposed pre-extracted SAM masks for this pair
        try:
            SAM_masks = np.load(f"{self.masks_dir}/{take_id2}/{cam2}/{vid_idx2}_masks.npz")
            SAM_masks = torch.from_numpy(SAM_masks['arr_0'].astype(np.uint8))
        except FileNotFoundError:
            print(f"WARNING: Masks not found for UID {take_id2}/{cam2}/{vid_idx2} - skipping this sample")
            return None
        
        if len(SAM_masks.shape) < 3:
            SAM_masks = torch.zeros((1, self.h2, self.w2))
        N_masks, H_masks, W_masks = SAM_masks.shape
        if H_masks != self.h2 or W_masks != self.w2:
            SAM_masks = F.interpolate(SAM_masks.unsqueeze(0).float(), size=(self.h2, self.w2), mode='nearest').squeeze(0).long()
        
        # Get the adjacent matrix
        adj_matrix = get_adj_matrix(SAM_masks, order=self.order)
        
        try:
            SAM_bboxes_dest = np.load(f"{self.masks_dir}/{take_id2}/{cam2}/{vid_idx2}_boxes.npy")
            SAM_bboxes_dest = torch.from_numpy(SAM_bboxes_dest.astype(np.float32))
        except FileNotFoundError:
            print(f"WARNING: Bounding boxes not found for UID {take_id2}/{cam2}/{vid_idx2} - skipping this sample")
            return None
        
        h_factor = self.h2 / H_masks
        w_factor = self.w2 / W_masks
        if h_factor != 1 or w_factor != 1:
            SAM_bboxes_dest[:, 0] = SAM_bboxes_dest[:, 0] * w_factor
            SAM_bboxes_dest[:, 1] = SAM_bboxes_dest[:, 1] * h_factor
            SAM_bboxes_dest[:, 2] = SAM_bboxes_dest[:, 2] * w_factor
            SAM_bboxes_dest[:, 3] = SAM_bboxes_dest[:, 3] * h_factor       

        if self.train_mode:
            visible_pixels = mask2_GT.sum()
            if visible_pixels > 0:
                NEG_SAM_masks, NEG_SAM_bboxes = self.select_adjacent_negatives(adj_matrix, SAM_bboxes_dest, SAM_masks, mask2_GT)
                is_visible = torch.tensor(1.)
                POS_SAM_masks = mask2_GT
                POS_SAM_bboxes, _ = bbox_from_mask(mask2_GT)
            else:
                N_remaining_indices = self.N_masks_per_batch - 1
                if SAM_masks.shape[0] < N_remaining_indices:
                    random_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=True)
                else:
                    random_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=False)
                
                NEG_SAM_masks = SAM_masks[random_indices]
                NEG_SAM_bboxes = SAM_bboxes_dest[random_indices]
                is_visible = torch.tensor(0.)
                random_idx = np.random.randint(SAM_masks.shape[0])
                POS_SAM_masks = SAM_masks[random_idx]
                POS_SAM_bboxes = SAM_bboxes_dest[random_idx]

            POS_mask_position = random.randint(0, self.N_masks_per_batch - 1)
            NEG_part1 = NEG_SAM_masks[:POS_mask_position]
            NEG_part2 = NEG_SAM_masks[POS_mask_position:]
            DEST_SAM_masks = torch.cat((NEG_part1, POS_SAM_masks.unsqueeze(0), NEG_part2), dim=0)

            NEG_part1_bboxes = NEG_SAM_bboxes[:POS_mask_position]
            NEG_part2_bboxes = NEG_SAM_bboxes[POS_mask_position:]
            DEST_SAM_bboxes = torch.cat((NEG_part1_bboxes, POS_SAM_bboxes.unsqueeze(0), NEG_part2_bboxes), dim=0)
        else:
            DEST_SAM_masks = SAM_masks
            visible_pixels = mask2_GT.sum()
            if visible_pixels > 0:
                is_visible = torch.tensor(1.)
            else:
                is_visible = torch.tensor(0.)
            POS_mask_position = self.get_best_mask(SAM_masks, mask2_GT)
            DEST_SAM_bboxes = SAM_bboxes_dest
            if len(DEST_SAM_bboxes.shape) == 1:
                DEST_SAM_bboxes = torch.zeros((1, 4))
        
        return {
            # Precomputed features (instead of images)
            'SOURCE_features': SOURCE_features,  # [C, H, W] - raw DINO features (384-dim)
            'DEST_features': DEST_features,      # [C, H, W] - raw DINO features (384-dim)
            # Mask and bbox info
            'SOURCE_mask': mask_SOURCE, 
            'SOURCE_bbox': bbox_from_mask(mask_SOURCE)[0], 
            'SOURCE_img_size': torch.tensor([self.h1, self.w1]),
            'GT_mask': mask2_GT, 
            'DEST_SAM_masks': DEST_SAM_masks, 
            'DEST_SAM_bbox': DEST_SAM_bboxes, 
            'DEST_img_size': torch.tensor([self.h2, self.w2]),
            'is_visible': is_visible, 
            'POS_mask_position': POS_mask_position.clone().detach() if isinstance(POS_mask_position, torch.Tensor) else torch.tensor(POS_mask_position),
            'pair_idx': torch.tensor(idx_sample)
        }

