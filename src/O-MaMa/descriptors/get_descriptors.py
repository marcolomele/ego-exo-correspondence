""" This extracts DINO descriptors of each mask, and outputs the batch with positive and negative pairs """

import torch
import torch.nn as nn
import os

class DescriptorExtractor:
    def __init__(self, dino_model, patch_size, context_size, device):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load DINOv2 pre-trained model 
        # self.model = torch.hub.load('facebookresearch/dinov2', dino_model).to(device)

        # Load DINOv3 pre-trained model 
        dinov3_dir = os.path.join(current_dir, '..', '..', 'dinov3-main')
        dinov3_dir = os.path.abspath(dinov3_dir)
        weights_path = os.path.join(dinov3_dir, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        self.model = torch.hub.load(dinov3_dir, dino_model, source='local', weights=weights_path).to(device)
        self.model.eval()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            dummy_features = self.model.forward_features(dummy_input)
            dino_feat_dim = dummy_features['x_norm_patchtokens'].shape[-1]
        
        # Create projection layer to map from DINO feature dim to 768 (expected by model)
        target_feat_dim = 768
        if dino_feat_dim != target_feat_dim:
            self.feature_proj = nn.Linear(dino_feat_dim, target_feat_dim).to(device)
            # Initialize with identity-like transformation (scaled)
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
        else:
            self.feature_proj = None
        
        self.patch_size = patch_size
        self.device = device
        self.context_size = context_size

    def extract_dense_features(self, image_tensor):
        # Extract features with DINO model (frozen, no gradients)
        with torch.no_grad():
            B, _, h, w = image_tensor.shape
            features = self.model.forward_features(image_tensor)
            patch_tokens = features['x_norm_patchtokens'].detach()
            patch_tokens = patch_tokens.reshape((B, h // self.patch_size, w // self.patch_size, patch_tokens.shape[-1])).permute(0, 3, 1, 2)
        
        # Project features from DINO dimension (384) to target dimension (768) if needed
        # This projection is trainable (outside no_grad context)
        if self.feature_proj is not None:
            # Reshape to [B*H*W, C] for linear layer, then reshape back
            B, C, H, W = patch_tokens.shape
            patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(B * H * W, C)
            patch_tokens = self.feature_proj(patch_tokens)
            patch_tokens = patch_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return patch_tokens.to(self.device)
       
    def dense_mask(self, masks, features):
        B, Nmasks, _, _ = masks.shape
        descriptor_list = torch.zeros((B, Nmasks, features.shape[1]), device=features.device)
        for b in range(masks.shape[0]):
            masks_expanded_batch = torch.nn.functional.interpolate(masks[b].unsqueeze(0).float(),
                                                                size = (features.shape[2], features.shape[3]),
                                                                mode='nearest').squeeze(0).to(self.device)
            for m in range(masks_expanded_batch.shape[0]):
                mask_expanded = masks_expanded_batch[m].unsqueeze(0)
                mask_descriptor = features[b] * mask_expanded
                mask_sum = mask_expanded.sum(dim=(1, 2), keepdim=True)  # Nº of píxeles with active mask per channel
                if mask_sum != 0:
                    feature_mean = (mask_descriptor.sum(dim=(1, 2)) / mask_sum.squeeze(0)).nan_to_num(0)
                    descriptor_list[b, m] = feature_mean
        return descriptor_list
     
    def add_context_to_bbox(self, bboxes_masks, context_size, H_max, W_max, reduction_factor):
        bboxes_context = bboxes_masks.clone().to(torch.int32) # Format x1, y1, w, h
        
        #Convert to x1, y1, x2, y2
        bboxes_context[:, :, 2] = bboxes_context[:, :, 0] + bboxes_context[:, :, 2]
        bboxes_context[:, :, 3] = bboxes_context[:, :, 1] + bboxes_context[:, :, 3]

        # Add context and validate limits
        bboxes_context[:, :, 0] = torch.clamp(bboxes_context[:, :, 0] - context_size, 0, W_max)  # x1 - context
        bboxes_context[:, :, 1] = torch.clamp(bboxes_context[:, :, 1] - context_size, 0, H_max)  # y1 - context
        bboxes_context[:, :, 2] = torch.clamp(bboxes_context[:, :, 2] + context_size, 0, W_max)  # x2 + context
        bboxes_context[:, :, 3] = torch.clamp(bboxes_context[:, :, 3] + context_size, 0, H_max)  # y2 + context

        bboxes_context = torch.floor(bboxes_context / reduction_factor).int()
        bboxes_context[:, :, 2] = torch.max(bboxes_context[:, :, 0] + 1, bboxes_context[:, :, 2])
        bboxes_context[:, :, 3] = torch.max(bboxes_context[:, :, 1] + 1, bboxes_context[:, :, 3])

        # Limit coordinates to the grid
        max_x = W_max // reduction_factor - 1
        max_y = H_max // reduction_factor - 1
        bboxes_context[:, :, 0] = torch.clamp(bboxes_context[:, :, 0], 0, max_x)  # x1
        bboxes_context[:, :, 1] = torch.clamp(bboxes_context[:, :, 1], 0, max_y)  # y1
        bboxes_context[:, :, 2] = torch.clamp(bboxes_context[:, :, 2], 0, max_x + 1)  # x2
        bboxes_context[:, :, 3] = torch.clamp(bboxes_context[:, :, 3], 0, max_y + 1)  # y2
        return bboxes_context

    def dense_bbox(self, bboxes_masks, img_sizes, features, context_sizes, reduction_factor):
        _, C, _, _ = features.shape
        H_max = img_sizes[:, 0].max().item()
        W_max = img_sizes[:, 1].max().item()
        descriptors = []

        # Iterate through batches
        for b in range(features.shape[0]):
            batch_descriptors = []

            # Process each context size
            for context_size in context_sizes:
                # Add context to bounding boxes for the current size
                bboxes_context = self.add_context_to_bbox(bboxes_masks[b].unsqueeze(0), context_size, H_max, W_max, reduction_factor)

                # Initialize the descriptor list for the current context size
                descriptor_list = torch.zeros((bboxes_context.shape[1], C), device=features.device)
                # Iterate through bounding boxes
                for i in range(bboxes_context.shape[1]):
                    x1, y1, x2, y2 = bboxes_context[0, i, :]
                    mask_descriptors = features[b, :, int(y1.item()):int(y2.item()), int(x1.item()):int(x2.item())]
                    mean_descriptor = mask_descriptors.mean(dim=(1, 2))
                    
                    descriptor_list[i] = mean_descriptor
                    
                batch_descriptors.append(descriptor_list)

            # Concatenate descriptors for all context sizes for this batch
            descriptors.append(torch.cat(batch_descriptors, dim=1))

        # Stack all batch descriptors
        descriptor_list = torch.stack(descriptors)
        return descriptor_list
    
    # The DEST descriptors are N masks, with negatives and a positive pair
    def get_DEST_descriptors(self, batch):
        DEST_img = batch['GT_img'].to(self.device)

        feats_DEST_img = self.extract_dense_features(DEST_img)
        
        _, _, h, w = feats_DEST_img.shape
        reduction_factor = 4
        dense_features = torch.nn.functional.interpolate(feats_DEST_img, 
                                                            size = (int(h * self.patch_size / reduction_factor), int(w * self.patch_size / reduction_factor)), 
                                                            mode='bilinear', align_corners=False)
        context_DEST_descriptors = self.dense_bbox(batch['DEST_SAM_bbox'], batch['DEST_img_size'], dense_features, context_sizes=[100], reduction_factor=reduction_factor)
        mask_DEST_descriptors = self.dense_mask(batch['DEST_SAM_masks'], dense_features)
        DEST_descriptors = torch.cat((mask_DEST_descriptors, context_DEST_descriptors), dim=2).to(self.device)

        return DEST_descriptors.to(self.device), feats_DEST_img

    # The SOURCE descriptors are just one mask 
    def get_SOURCE_descriptors(self, batch):
        SOURCE_img = batch['SOURCE_img'].to(self.device)

        feats_SOURCE_img = self.extract_dense_features(SOURCE_img)

        _, _, h, w = feats_SOURCE_img.shape
        reduction_factor = 4
        dense_features = torch.nn.functional.interpolate(feats_SOURCE_img, 
                                                            size = (int(h * self.patch_size / reduction_factor), int(w * self.patch_size / reduction_factor)), 
                                                            mode='bilinear', align_corners=False)
        context_SOURCE_descriptors = self.dense_bbox(batch['SOURCE_bbox'].unsqueeze(1), batch['SOURCE_img_size'], dense_features, context_sizes=[100], reduction_factor=reduction_factor)
        mask_SOURCE_descriptors = self.dense_mask(batch['SOURCE_mask'].unsqueeze(1), dense_features)
        SOURCE_descriptors = torch.cat((mask_SOURCE_descriptors, context_SOURCE_descriptors), dim=2).to(self.device)
        
        return SOURCE_descriptors.to(self.device), feats_SOURCE_img
    