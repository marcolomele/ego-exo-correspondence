""" This extracts DINO descriptors of each mask, and outputs the batch with positive and negative pairs """

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import timm

#create a ResNet model that outputs multi-layer features
class ResNetMultiLayer(nn.Module):
    """
    A multi-layer ResNet feature extractor with CLIP pre-training.
    This module extracts features from multiple layers of a ResNet-50 backbone 
    pre-trained with CLIP, projects them to a common channel dimension, and 
    fuses them into a single feature map.
    Attributes:
        resnet: A ResNet-50 model with CLIP pre-training, configured to extract 
                features from multiple layers.
        proj_layers: A ModuleList of 1x1 convolutional layers that project each 
                     feature map to 192 channels.
    Methods:
        forward(x): Processes input through the network.
            Args:
                x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is 
                                batch size, C is channels (typically 3), and H, W 
                                are spatial dimensions.
            Returns:
                torch.Tensor: Fused and downsampled feature map of shape 
                             (B, 768, 16, 16). The 768 channels result from 
                             concatenating 4 layers of 192 channels each.
            Processing steps:
                1. Extracts features from the last 4 ResNet layers
                2. Projects each feature map to 192 channels using 1x1 convolutions
                3. Upsamples all features to match the highest resolution
                4. Concatenates features along the channel dimension
                5. Downsamples the fused features to 16x16 spatial resolution
    """
    def __init__(self):
        super().__init__()
        # Load CLIP-pretrained ResNet backbone
        resnet = timm.create_model('resnet50_clip', pretrained=True, features_only=True)
        self.resnet = resnet

        # Keep only the last 4 layers
        channels = resnet.feature_info.channels()[1:]
        # Match projection layers to these channels
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(ch, 192, 1) for ch in channels
        ])

    def forward(self, x):
        # Extract intermediate feature maps
        feats = self.resnet(x)[1:]

        # Project each feature map to 192 channels
        for i in range(len(feats)):
            feats[i] = self.proj_layers[i](feats[i])

        # Align spatial sizes to the highest resolution
        target_size = feats[0].shape[2:]
        feats = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                 for f in feats]

        # Fuse and downsample
        fused = torch.cat(feats, dim=1)
        fused_down = F.interpolate(fused, size=(16, 16), mode='bilinear', align_corners=False)
        return fused_down

class DescriptorExtractor:

    def __init__(self, context_size, device, dino_model=None, patch_size=None):
        # Load ResNet 
        self.model = ResNetMultiLayer().to(device)
        self.model.eval()
        self.device = device
        self.context_size = context_size

    def extract_dense_features(self, image_tensor):
        with torch.no_grad():  # Disable gradient computation
            features = self.model(image_tensor)
        return features.to(self.device) 
       
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
        dense_features = feats_DEST_img
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
        dense_features = feats_SOURCE_img
        context_SOURCE_descriptors = self.dense_bbox(batch['SOURCE_bbox'].unsqueeze(1), batch['SOURCE_img_size'], dense_features, context_sizes=[100], reduction_factor=reduction_factor)
        mask_SOURCE_descriptors = self.dense_mask(batch['SOURCE_mask'].unsqueeze(1), dense_features)
        SOURCE_descriptors = torch.cat((mask_SOURCE_descriptors, context_SOURCE_descriptors), dim=2).to(self.device)
        
        return SOURCE_descriptors.to(self.device), feats_SOURCE_img