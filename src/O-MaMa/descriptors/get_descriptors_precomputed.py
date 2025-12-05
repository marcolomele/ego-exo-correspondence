"""
Descriptor extractor for precomputed DINOv3 features.

This is a modified version of get_descriptors.py that:
- Does NOT load the DINO model
- Uses precomputed features loaded from the dataset
- Keeps the trainable projection layer (384 -> 768)
- Keeps dense_mask() and dense_bbox() methods unchanged
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorExtractorPrecomputed:
    """
    Descriptor extractor that uses precomputed DINOv3 features.
    
    This eliminates the DINO forward pass bottleneck while keeping the
    projection layer (384 -> 768) trainable.
    """
    
    def __init__(self, patch_size, context_size, device, dino_feat_dim=384, target_feat_dim=768):
        """
        Args:
            patch_size: Patch size of the DINO transformer
            context_size: Size of the context for bbox features
            device: PyTorch device
            dino_feat_dim: Feature dimension from DINO (384 for ViT-Small, 2048 for ResNet-50)
            target_feat_dim: Target feature dimension expected by model (768)
        """
        self.patch_size = patch_size
        self.device = device
        self.context_size = context_size
        self.dino_feat_dim = dino_feat_dim
        self.target_feat_dim = target_feat_dim
        
        # Trainable projection layers for different feature extractors
        # Both use 1x1 convolutions to preserve spatial structure and train efficiently
        if dino_feat_dim == 384:
            # DINOv3 ViT-S/14: 384 -> 768
            self.feature_proj = nn.Conv2d(384, 768, kernel_size=1, bias=True).to(device)
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
        elif dino_feat_dim == 2048:
            # ResNet-50: 2048 -> 768
            self.feature_proj = nn.Conv2d(2048, 768, kernel_size=1, bias=True).to(device)
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
        elif dino_feat_dim == target_feat_dim:
            # No projection needed
            self.feature_proj = None
        else:
            raise ValueError(f"Unsupported dino_feat_dim={dino_feat_dim}. Supported: 384 (DINOv3), 2048 (ResNet-50), or {target_feat_dim} (no projection)")
    
    def project_features(self, features):
        """
        Apply trainable projection to precomputed features.
        
        Args:
            features: [B, C, H, W] tensor of precomputed features (384-dim for DINOv3, 2048-dim for ResNet-50)
            
        Returns:
            projected: [B, C', H, W] tensor of projected features (768-dim)
        """
        features = features.to(self.device)
        
        if self.feature_proj is not None:
            # Apply Conv2d projection (preserves spatial structure)
            features = self.feature_proj(features)
        
        return features
       
    def dense_mask(self, masks, features):
        """
        Compute mask descriptors by averaging features within each mask.
        
        Args:
            masks: [B, N_masks, H, W] binary masks
            features: [B, C, H', W'] feature maps
            
        Returns:
            descriptors: [B, N_masks, C] mask descriptors
        """
        B, Nmasks, _, _ = masks.shape
        descriptor_list = torch.zeros((B, Nmasks, features.shape[1]), device=features.device)
        
        for b in range(masks.shape[0]):
            masks_expanded_batch = F.interpolate(
                masks[b].unsqueeze(0).float(),
                size=(features.shape[2], features.shape[3]),
                mode='nearest'
            ).squeeze(0).to(self.device)
            
            for m in range(masks_expanded_batch.shape[0]):
                mask_expanded = masks_expanded_batch[m].unsqueeze(0)
                mask_descriptor = features[b] * mask_expanded
                mask_sum = mask_expanded.sum(dim=(1, 2), keepdim=True)
                # Prevent division by zero - use max(mask_sum, 1) to avoid NaN
                mask_sum_safe = torch.clamp(mask_sum, min=1.0)
                feature_mean = (mask_descriptor.sum(dim=(1, 2)) / mask_sum_safe.squeeze(0))
                descriptor_list[b, m] = feature_mean
                    
        return descriptor_list
     
    def add_context_to_bbox(self, bboxes_masks, context_size, H_max, W_max, reduction_factor):
        """Add context padding to bounding boxes."""
        bboxes_context = bboxes_masks.clone().to(torch.int32)  # Format x1, y1, w, h
        
        # Convert to x1, y1, x2, y2
        bboxes_context[:, :, 2] = bboxes_context[:, :, 0] + bboxes_context[:, :, 2]
        bboxes_context[:, :, 3] = bboxes_context[:, :, 1] + bboxes_context[:, :, 3]

        # Add context and validate limits
        bboxes_context[:, :, 0] = torch.clamp(bboxes_context[:, :, 0] - context_size, 0, W_max)
        bboxes_context[:, :, 1] = torch.clamp(bboxes_context[:, :, 1] - context_size, 0, H_max)
        bboxes_context[:, :, 2] = torch.clamp(bboxes_context[:, :, 2] + context_size, 0, W_max)
        bboxes_context[:, :, 3] = torch.clamp(bboxes_context[:, :, 3] + context_size, 0, H_max)

        bboxes_context = torch.floor(bboxes_context / reduction_factor).int()
        bboxes_context[:, :, 2] = torch.max(bboxes_context[:, :, 0] + 1, bboxes_context[:, :, 2])
        bboxes_context[:, :, 3] = torch.max(bboxes_context[:, :, 1] + 1, bboxes_context[:, :, 3])

        # Limit coordinates to the grid
        max_x = W_max // reduction_factor - 1
        max_y = H_max // reduction_factor - 1
        bboxes_context[:, :, 0] = torch.clamp(bboxes_context[:, :, 0], 0, max_x)
        bboxes_context[:, :, 1] = torch.clamp(bboxes_context[:, :, 1], 0, max_y)
        bboxes_context[:, :, 2] = torch.clamp(bboxes_context[:, :, 2], 0, max_x + 1)
        bboxes_context[:, :, 3] = torch.clamp(bboxes_context[:, :, 3], 0, max_y + 1)
        
        return bboxes_context

    def dense_bbox(self, bboxes_masks, img_sizes, features, context_sizes, reduction_factor):
        """
        Compute bbox descriptors by averaging features within context-expanded bboxes.
        
        Args:
            bboxes_masks: [B, N_masks, 4] bounding boxes (x1, y1, w, h)
            img_sizes: [B, 2] image sizes (H, W)
            features: [B, C, H', W'] feature maps
            context_sizes: List of context sizes to use
            reduction_factor: Factor to reduce bbox coordinates
            
        Returns:
            descriptors: [B, N_masks, C * len(context_sizes)] bbox descriptors
        """
        _, C, _, _ = features.shape
        H_max = img_sizes[:, 0].max().item()
        W_max = img_sizes[:, 1].max().item()
        descriptors = []

        for b in range(features.shape[0]):
            batch_descriptors = []

            for context_size in context_sizes:
                bboxes_context = self.add_context_to_bbox(
                    bboxes_masks[b].unsqueeze(0), context_size, H_max, W_max, reduction_factor
                )

                descriptor_list = torch.zeros((bboxes_context.shape[1], C), device=features.device)
                
                for i in range(bboxes_context.shape[1]):
                    x1, y1, x2, y2 = bboxes_context[0, i, :]
                    mask_descriptors = features[b, :, int(y1.item()):int(y2.item()), int(x1.item()):int(x2.item())]
                    # Handle empty bboxes - use nanmean or replace NaN with zeros
                    if mask_descriptors.numel() > 0:
                        mean_descriptor = mask_descriptors.mean(dim=(1, 2)).nan_to_num(0.0)
                    else:
                        mean_descriptor = torch.zeros(C, device=features.device)
                    descriptor_list[i] = mean_descriptor
                    
                batch_descriptors.append(descriptor_list)

            descriptors.append(torch.cat(batch_descriptors, dim=1))

        descriptor_list = torch.stack(descriptors)
        return descriptor_list
    
    def get_DEST_descriptors(self, batch):
        """
        Get DEST descriptors from precomputed features.
        
        Args:
            batch: Dictionary containing 'DEST_features', 'DEST_SAM_bbox', 
                   'DEST_img_size', 'DEST_SAM_masks'
                   
        Returns:
            DEST_descriptors: [B, N_masks, C*2] descriptors (mask + context)
            feats_DEST_img: [B, C, H, W] projected feature maps
        """
        # Load precomputed features and apply projection
        DEST_features = batch['DEST_features'].to(self.device)
        feats_DEST_img = self.project_features(DEST_features)
        
        _, _, h, w = feats_DEST_img.shape
        reduction_factor = 4
        dense_features = F.interpolate(
            feats_DEST_img, 
            size=(int(h * self.patch_size / reduction_factor), int(w * self.patch_size / reduction_factor)), 
            mode='bilinear', 
            align_corners=False
        )
        
        context_DEST_descriptors = self.dense_bbox(
            batch['DEST_SAM_bbox'], 
            batch['DEST_img_size'], 
            dense_features, 
            context_sizes=[100], 
            reduction_factor=reduction_factor
        )
        mask_DEST_descriptors = self.dense_mask(batch['DEST_SAM_masks'], dense_features)
        DEST_descriptors = torch.cat((mask_DEST_descriptors, context_DEST_descriptors), dim=2).to(self.device)

        return DEST_descriptors, feats_DEST_img

    def get_SOURCE_descriptors(self, batch):
        """
        Get SOURCE descriptors from precomputed features.
        
        Args:
            batch: Dictionary containing 'SOURCE_features', 'SOURCE_bbox', 
                   'SOURCE_img_size', 'SOURCE_mask'
                   
        Returns:
            SOURCE_descriptors: [B, 1, C*2] descriptors (mask + context)
            feats_SOURCE_img: [B, C, H, W] projected feature maps
        """
        # Load precomputed features and apply projection
        SOURCE_features = batch['SOURCE_features'].to(self.device)
        feats_SOURCE_img = self.project_features(SOURCE_features)

        _, _, h, w = feats_SOURCE_img.shape
        reduction_factor = 4
        dense_features = F.interpolate(
            feats_SOURCE_img, 
            size=(int(h * self.patch_size / reduction_factor), int(w * self.patch_size / reduction_factor)), 
            mode='bilinear', 
            align_corners=False
        )
        
        context_SOURCE_descriptors = self.dense_bbox(
            batch['SOURCE_bbox'].unsqueeze(1), 
            batch['SOURCE_img_size'], 
            dense_features, 
            context_sizes=[100], 
            reduction_factor=reduction_factor
        )
        mask_SOURCE_descriptors = self.dense_mask(batch['SOURCE_mask'].unsqueeze(1), dense_features)
        SOURCE_descriptors = torch.cat((mask_SOURCE_descriptors, context_SOURCE_descriptors), dim=2).to(self.device)
        
        return SOURCE_descriptors, feats_SOURCE_img

