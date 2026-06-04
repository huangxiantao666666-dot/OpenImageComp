"""
Script Name: simopa_ext.py (Extended SimOPA)
Purpose: Object placement assessment model that leverages reference objects in the background.

This model extends the basic SimOPA by incorporating information from reference objects
(e.g., their bounding boxes, features, masks) to reason about the rationality of foreground placement.
The intuition is that object placement should be judged not in isolation but in relation to
existing objects in the scene (e.g., a chair should be placed next to a table, not floating in mid-air).

Key features:
- Supports 6 different relation feature fusion methods (relation_method 0-5)
- Supports 3 different attention weighting mechanisms (attention_method 0-2)
- Uses ROI Align to extract region features from feature maps
- Can incorporate geometric features (relative position/size) and mask features

Inputs:
- img_cat: [B, 4, 256, 256] composite image + mask
- target_box: [B, 4] bounding box of foreground object
- refer_box: [B, N, 6] bounding boxes of N reference objects
- target_feature: [B, 1, 2048] pre-extracted target features
- refer_feature: [B, N, 2048] pre-extracted reference features
- target_mask: [B, 1, 64, 64] foreground mask
- refer_mask: [B, N, 64, 64] reference object masks
- w, h: [B] original image dimensions for coordinate scaling
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import torchvision
import numpy as np
import os, sys

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from resnet_4ch import resnet
from simopa_ext_config import opt
from PIL import Image
import torchvision.transforms as transforms
import cv2
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ROI ALIGN FUNCTION
# ============================================================================
def roi_align(feature_map, boxes, w, h, outsize=opt.roi_align_size, insize=opt.global_feature_size):
    """
    Extract region features from a feature map using ROI Align.
    
    This function takes bounding boxes in original image coordinates, scales them
    to the feature map size, and extracts fixed-size feature regions.
    
    Args:
        feature_map: [B, C, insize, insize] feature map (insize=8 from config), layer4 output of resnet18 is 32x downsample
        boxes: [B, N, 4] bounding boxes in original image coordinates
        w, h: [B] original image width and height
        outsize: output size of pooled regions (default 3 from config)
        insize: feature map size (default 8 from config)
    
    Returns:
        pooled_regions: [B*N, C, outsize, outsize] features for each box
    """
    boxes_ = boxes.clone()
    
    # Add a dummy dimension if boxes is 2D [B,4] -> [B,1,4]
    if boxes_.dim() == 2:
        boxes_ = boxes_.unsqueeze(1)
    
    B, N, _ = boxes_.shape
    
    # Scale bounding boxes from original image coordinates to feature map coordinates
    # Feature map size is insize (e.g., 8x8), original image is w x h
    scaled_boxes = torch.zeros_like(boxes_)
    scaled_boxes[:, :, 0::2] = boxes_[:, :, 0::2] * (insize / w[:, None, None]).int()  # x coordinates, int makes it a ROI pooling
    scaled_boxes[:, :, 1::2] = boxes_[:, :, 1::2] * (insize / h[:, None, None]).int()  # y coordinates
    
    # Create batch indices: [0,0,0,...,1,1,1,...,B-1,...]
    batch_index = torch.arange(B).view(-1, 1).repeat(1, N).reshape(B, N, 1).to(boxes_.device)
    batch_index = batch_index.float()
    
    # Concatenate batch indices with scaled boxes to form ROIs
    # ROI format: [batch_index, x1, y1, x2, y2] (required by torchvision roi_align)
    rois = torch.cat((batch_index, scaled_boxes), dim=-1) # [B, N, 5]
    rois = rois.view(B * N, -1)  # [B*N, 5]
    
    # Apply ROI Pooling (Since it use int()) to extract fixed-size region features
    pooled_regions = torchvision.ops.roi_align(feature_map, rois,
                                               output_size=(outsize, outsize))
    return pooled_regions  # [B*N, C, outsize, outsize]


# ============================================================================
# SELF-ATTENTION MODULE
# ============================================================================
class SelfAttention(nn.Module):
    """
    Multi-head self-attention module for computing relationships between objects.
    
    Computes pairwise attention scores between N items (e.g., reference objects).
    Used to determine which reference objects are most relevant for judging the foreground.
    
    Args:
        dim: Input feature dimension (region_feature_dim)
        heads: Number of attention heads (16 from config)
        dim_head: Dimension per head (64 from config)
    
    Output:
        attn: [B, N, heads, N] attention scores for each head and each pair of objects
    """
    def __init__(self, dim, heads=16, dim_head=64):
        super(SelfAttention, self).__init__()
        inner_dim = dim_head * heads  # 64 * 16 = 1024
        self.norm = nn.LayerNorm(dim)  # Layer normalization for stable training
        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for dot product (1/sqrt(64))
        
        # Linear layer to generate query and key (no bias)
        # Output dimension: inner_dim * 2 (half for query, half for key)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, N, dim] features for N objects
        
        Returns:
            attn: [B, N, heads, N] attention scores
        """
        x = self.norm(x)  # [B, N, dim]
        b, n, _, h = *x.shape, self.heads  # b=batch, n=N, h=heads
        
        # Generate query and key, split into two tensors of [B, N, inner_dim]
        qk = self.to_qk(x).chunk(2, dim=-1)
        
        # Reshape: [B, N, inner_dim] -> [B, heads, N, dim_head]
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qk)
        
        # Compute attention scores: Q @ K^T, scaled
        # einsum: b h i d, b h j d -> b h i j
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Rearrange to [B, N, heads, N] for easier processing
        attn = rearrange(dots, 'b h n j -> b n h j')
        return attn


# ============================================================================
# CUSTOM BOTTLENECK BLOCK (For geometric feature extraction)
# ============================================================================
class _Bottleneck(nn.Module):
    """
    Custom bottleneck block with expansion=2 (standard ResNet uses expansion=4).
    
    Structure: Conv1x1 (reduce) -> Conv3x3 -> Conv1x1 (expand to 2x)
    
    Used in the geometric feature extraction network for mask processing.
    """
    expansion = 2  # Output channels = planes * 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Bottleneck, self).__init__()
        # 1x1 convolution: reduce dimension
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 convolution: spatial processing (may downsample)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 convolution: expand dimension to planes * expansion (2x)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Optional downsampling for residual path
        self.stride = stride

    def forward(self, x):
        residual = x

        # First conv block (reduce)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (spatial)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Third conv block (expand)
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply downsampling to residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)

        # Skip connection
        out += residual
        out = self.relu(out)

        return out


# ============================================================================
# MAIN MODEL: ObjectPlaceNet (Extended SimOPA)
# ============================================================================
class ObjectPlaceNet(nn.Module):
    """
    Extended SimOPA model for object placement assessment with reference objects.
    
    This model can operate in two modes:
    1. relation_method = None: Standard classification (like basic SimOPA)
    2. relation_method in [0,1,2,3,4,5]: Uses reference objects for relational reasoning
    
    The model extracts:
    - Global features from the entire composite image
    - Region features from foreground and reference objects
    - Optional geometric features (relative positions/sizes)
    - Optional mask features (shape information)
    
    Features are then fused using attention mechanisms to produce the final prediction.
    """
    
    def __init__(self, backbone_pretrained=True):
        """
        Initialize the Extended SimOPA model.
        
        Args:
            backbone_pretrained: Whether to load pretrained ResNet weights
        """
        super(ObjectPlaceNet, self).__init__()
        
        # ====================================================================
        # BACKBONE: Modified ResNet for 4-channel input
        # ====================================================================
        resnet_layers = int(opt.backbone.split('resnet')[-1])  # e.g., 'resnet18' -> 18
        
        if backbone_pretrained:
            # Load with pretrained weights
            backbone = resnet(resnet_layers,
                              backbone_pretrained,
                              os.path.join(opt.pretrained_model_path, opt.backbone + '.pth'))
        else:
            # Load without pretrained weights
            backbone = resnet(resnet_layers, opt.without_mask)
        
        # Remove the final two layers (avgpool and fc) to get feature extractor only
        # Keep conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone
        
        # Global feature dimension depends on backbone
        # ResNet18/34: 512 channels, ResNet50/101/152: 2048 channels
        self.global_feature_dim = 512 if opt.backbone in ['resnet18', 'resnet34'] else 2048
        
        # Simple classifier for baseline mode (no reference objects)
        if opt.relation_method is None:
            self.fc_global = nn.Linear(self.global_feature_dim,
                                       opt.class_num, bias=False)
        
        # ====================================================================
        # GEOMETRIC FEATURE EXTRACTOR (for relation_method == 5)
        # ====================================================================
        if opt.relation_method == 5: # 5 times forward
            # Small CNN to extract features from mask pairs (target mask + reference mask)
            # Input: 2-channel mask (target + reference), size 64x64
            # Output: geometric_feature_dim (256) feature vector
            self.geometric_layers = nn.Sequential(
                # First conv block: 2 -> 64, stride=2 (32x32)
                nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16x16
                
                # Second conv block: 64 -> 256, stride=2 (8x8)
                nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 4x4, 4x4x256
                
                # Flatten and project to geometric_feature_dim
                nn.Flatten(1),
                nn.Linear((opt.binary_mask_size // 16) ** 2 * 256,  # (64/16=4) -> 4*4*256=4096
                          opt.geometric_feature_dim, bias=False)    # 4096 -> 256
            )
        
        # ====================================================================
        # POOLING LAYERS
        # ====================================================================
        self.avgpool3x3 = nn.AdaptiveAvgPool2d(3)  # For region features (unused in base config)
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)  # For global features
        
        # ====================================================================
        # RELATION INFRASTRUCTURE (when reference objects are used)
        # ====================================================================
        if opt.relation_method is not None:
            self.concatenate_dim = 1024  # Base dimension for feature concatenation
            
            # Different relation methods have different feature extraction strategies
            if opt.relation_method == 0:
                # Method 0: ROI Align only
                # Extract region features directly from feature map using ROI Align
                self.roi_align = roi_align
                self.roi_feature = nn.Linear(
                    self.global_feature_dim * opt.roi_align_size * opt.roi_align_size,  # 512 * 3 * 3 = 4608
                    512, bias=False
                )
                self.region_feature_dim = 1024
                self.fc_region_feature = nn.Linear(self.concatenate_dim, self.region_feature_dim, bias=False)
                
            elif opt.relation_method in [1, 2]:
                # Method 1: Use pre-extracted target/refer features directly
                # Method 2: Average of target and refer features
                self.region_feature_dim = 2048  # Pre-extracted feature dimension
                
            else:
                # Methods 3, 4, 5: ROI Feature extraction with additional information
                self.roi_feature = nn.Linear(2048, 512, bias=False)  # Compress pre-extracted features
                
                if opt.relation_method == 3:
                    # Method 3: Only ROI features (no geometry)
                    self.concatenate_dim = 1024  # 512 (target) + 512 (refer) = 1024
                    
                elif opt.relation_method == 4:
                    # Method 4: ROI features + simple geometry (8-dim)
                    # 8 dimensions: [refer_x, refer_y, refer_w, refer_h, target_x, target_y, target_w, target_h]
                    self.concatenate_dim += 8  # 1024 + 8 = 1032
                    
                else:  # method == 5
                    # Method 5: ROI features + mask-derived geometry (256-dim)
                    self.concatenate_dim += opt.geometric_feature_dim  # 1024 + 256 = 1280
                
                # Determine region feature dimension (cap at 1024)
                if self.concatenate_dim > 1024:
                    self.region_feature_dim = 1024
                else:
                    self.region_feature_dim = self.concatenate_dim
                
                self.fc_region_feature = nn.Linear(self.concatenate_dim, self.region_feature_dim, bias=False)
        
        # ====================================================================
        # ATTENTION INFRASTRUCTURE
        # ====================================================================
        # Self-attention module for methods 0 and 2
        if opt.attention_method in [0, 2]:
            self.refer_attention = SelfAttention(
                self.region_feature_dim,
                heads=opt.attention_head,        # 16
                dim_head=opt.attention_dim_head  # 64
            )
        
        # Weight learning layers for different attention methods
        if opt.attention_method == 0:
            # Method 0: Learn weights from attention scores only
            self.fc_weight_learn = nn.Linear(opt.attention_head, 1, bias=False)  # 16 -> 1
        elif opt.attention_method == 1:
            # Method 1: Learn weights from region features directly
            self.fc_weight_learn = nn.Linear(self.region_feature_dim, 1, bias=False)
        elif opt.attention_method == 2:
            # Method 2: Learn weights from concatenated region features and attention scores
            self.fc_weight_learn = nn.Linear(opt.attention_head + self.region_feature_dim, 1, bias=False)
        
        # ====================================================================
        # PREDICTION HEAD
        # ====================================================================
        if opt.relation_method is None:
            # Baseline mode: single linear layer
            self.prediction_head = nn.Linear(self.global_feature_dim, opt.class_num, bias=False)
        else:
            # Advanced mode: MLP with dropout
            fusion_feature_dim = self.region_feature_dim + self.global_feature_dim if not opt.without_global_feature else self.region_feature_dim
            
            self.prediction_head = nn.Sequential(
                nn.Linear(fusion_feature_dim, fusion_feature_dim, bias=False),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(fusion_feature_dim, 512, bias=False),
                nn.ReLU(True),
                nn.Linear(512, opt.class_num, bias=False)
            )
    
    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    def forward(self, img_cat, target_box, refer_box, target_feature, refer_feature, target_mask, refer_mask, w, h):
        """
        Forward pass of the Extended SimOPA model.
        
        Args:
            img_cat: [B, 4, 256, 256] Composite image (RGB) + mask (grayscale) concatenated
            target_box: [B, 4] Foreground object bounding box [x1, y1, x2, y2]
            refer_box: [B, N, 6] Reference object bounding boxes (N=refer_num=5)
            target_feature: [B, 1, 2048] Pre-extracted features for foreground
            refer_feature: [B, N, 2048] Pre-extracted features for reference objects
            target_mask: [B, 1, 64, 64] Foreground mask
            refer_mask: [B, N, 64, 64] Reference object masks
            w, h: [B] Original image width and height
        
        Returns:
            prediction: [B, 2] Logits for reasonable/unreasonable
            attention_weights: [B, N] Attention weights for each reference object (or None)
        """
        batch_size = img_cat.shape[0]
        global_feature = None
        
        # --------------------------------------------------------------------
        # STEP 1: Extract global features from the entire image
        # --------------------------------------------------------------------
        if opt.without_mask:
            # Discard mask channel if configured (use only RGB)
            img_cat = img_cat[:, 0:3]
        
        # Pass through backbone to get feature map
        feature_map = self.backbone(img_cat)  # [B, C, 8, 8] for ResNet18 (C=512)
        
        # Global average pooling to get global feature vector
        global_feature = self.avgpool1x1(feature_map)  # [B, C, 1, 1]
        global_feature = global_feature.flatten(1)     # [B, C]
        
        # --------------------------------------------------------------------
        # STEP 2: If no relation method, return global classification
        # --------------------------------------------------------------------
        if opt.relation_method is None:
            prediction = self.prediction_head(global_feature)
            return prediction
        
        # --------------------------------------------------------------------
        # STEP 3: Prepare bounding boxes for ROI operations
        # --------------------------------------------------------------------
        refer_boxes = refer_box[:, :, :4]      # [B, N, 4] - take first 4 columns as coordinates
        target_boxes = target_box[:, None, :]  # [B, 1, 4] - add dimension for broadcasting
        
        # --------------------------------------------------------------------
        # STEP 4: Extract region features (varies by relation_method)
        # --------------------------------------------------------------------
        region_feature = None
        
        if opt.relation_method == 0:
            # Method 0: Extract region features using ROI Align
            refer_feature = self.roi_align(feature_map, refer_boxes, w, h)  # [B*N, C, 3, 3]
            refer_feature = self.roi_feature(refer_feature.flatten(1))      # [B*N, Cx3x3] -> [B*N, 512]
            refer_feature = refer_feature.view(batch_size, opt.refer_num, -1)  # [B, N, 512]
            
            target_feature = self.roi_align(feature_map, target_boxes, w, h)   # [B, C, 3, 3]
            target_feature = self.roi_feature(target_feature.flatten(1))       # [B, 512]
            target_feature = target_feature[:, None, :].repeat(1, opt.refer_num, 1)  # [B, N, 512]
            
            # Concatenate refer and target features
            region_feature = torch.cat((refer_feature, target_feature), dim=-1)  # [B, N, 1024]
            region_feature = self.fc_region_feature(region_feature)  # [B, N, 1024]
            
        elif opt.relation_method in [1, 2]:
            # Method 1: Use pre-extracted features directly
            # for ablation study that if reference feature is useful
            if opt.relation_method == 1:
                region_feature = target_feature  # [B, 1, 2048] -> [B, N, 2048]? Broadcasting handled
            else:
                # Method 2: Average of refer and target features\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
                region_feature = torch.mean(torch.cat([refer_feature, target_feature], dim=1), dim=1, keepdim=True)
                
        elif opt.relation_method in [3, 4, 5]:
            # Methods 3-5: Compress pre-extracted features first
            refer_feature = self.roi_feature(refer_feature)  # [B, N, 2048] -> [B, N, 512]
            target_feature = self.roi_feature(target_feature)  # [B, 1, 2048] -> [B, 1, 512]
            target_feature = target_feature.repeat(1, opt.refer_num, 1)  # [B, N, 512]
            
            # Base concatenation of refer and target features
            region_feature = torch.cat([refer_feature, target_feature], dim=2)  # [B, N, 1024]
            
            if opt.relation_method == 3:
                # Method 3: Only ROI features (no additional info)
                region_feature = self.fc_region_feature(region_feature)  # [B, N, 1024]
                
            elif opt.relation_method == 4:
                # Method 4: Add simple geometric features (normalized coordinates)
                exp_w, exp_h = w.unsqueeze(1), h.unsqueeze(1)  # [B, 1]
                
                # Reference object geometry (normalized)
                refer_x = (refer_boxes[:, :, 0] + refer_boxes[:, :, 2]) / (2 * exp_w)  # center x
                refer_y = (refer_boxes[:, :, 1] + refer_boxes[:, :, 3]) / (2 * exp_h)  # center y
                refer_w = (refer_boxes[:, :, 2] - refer_boxes[:, :, 0]) / exp_w        # width
                refer_h = (refer_boxes[:, :, 3] - refer_boxes[:, :, 1]) / exp_h        # height
                
                # Target object geometry (normalized, repeated N times)
                target_x = (target_boxes[:, :, 0] + target_boxes[:, :, 2]) / (2 * exp_w).repeat(1, opt.refer_num)
                target_y = (target_boxes[:, :, 1] + target_boxes[:, :, 3]) / (2 * exp_h).repeat(1, opt.refer_num)
                target_w = (target_boxes[:, :, 2] - target_boxes[:, :, 0]) / exp_w.repeat(1, opt.refer_num)
                target_h = (target_boxes[:, :, 3] - target_boxes[:, :, 1]) / exp_h.repeat(1, opt.refer_num)
                
                # Stack geometric features: 8 dimensions total
                geometric_feature = torch.stack([
                    refer_x, refer_y, refer_w, refer_h,
                    target_x, target_y, target_w, target_h
                ], dim=2)  # [B, N, 8]
                
                # Concatenate with ROI features
                fuse_feature = torch.cat([region_feature, geometric_feature], dim=-1)  # [B, N, 1032]
                region_feature = self.fc_region_feature(fuse_feature)  # [B, N, 1024]
                
            else:  # method == 5
                # Method 5: Add mask-derived geometric features
                mask_size = opt.binary_mask_size  # 64
                
                # Reshape masks for batch processing
                target_mask = target_mask.repeat(1, opt.refer_num, 1, 1).view(
                    batch_size * opt.refer_num, 1, mask_size, mask_size)  # [B*N, 1, 64, 64]
                refer_mask = refer_mask.view(batch_size * opt.refer_num, 1, mask_size, mask_size)  # [B*N, 1, 64, 64]
                
                # Concatenate target and refer masks (2-channel input)
                concat_mask = torch.cat([refer_mask, target_mask], dim=1)  # [B*N, 2, 64, 64]
                
                # Extract geometric features from mask pair
                geometric_feature = self.geometric_layers(concat_mask)  # [B*N, 256]
                geometric_feature = geometric_feature.view(batch_size, opt.refer_num, -1)  # [B, N, 256]
                
                # Concatenate ROI features with geometric features
                fused_feature = torch.cat((region_feature, geometric_feature), dim=2)  # [B, N, 1280]
                region_feature = self.fc_region_feature(fused_feature)  # [B, N, 1024]
        
        # --------------------------------------------------------------------
        # STEP 5: Apply attention weighting to aggregate reference features
        # --------------------------------------------------------------------
        agg_region_feature = None
        attention_weights = None
        
        if opt.attention_method is None:
            # No attention: simple average over reference objects
            agg_region_feature = torch.mean(region_feature, dim=1)  # [B, region_feature_dim]
            
        elif opt.attention_method == 1:
            # Method 1: Learn attention weights directly from region features
            attention_weights = self.fc_weight_learn(region_feature)  # [B, N, 1]
            attention_weights = F.softmax(attention_weights, dim=1)   # [B, N, 1]
            agg_region_feature = torch.sum(attention_weights * region_feature, dim=1)  # [B, region_feature_dim]
            
        else:  # attention_method == 0 or 2
            # Compute pairwise attention scores between reference objects
            similarity_vector = self.refer_attention(region_feature)  # [B, N, heads, N]
            similarity_vector = torch.mean(similarity_vector, dim=-1)  # [B, N, heads]
            
            if opt.attention_method == 0:
                # Method 0: Learn weights from attention scores only
                attention_weights = self.fc_weight_learn(similarity_vector)  # [B, N, 1]
                attention_weights = F.softmax(attention_weights, dim=1)
                agg_region_feature = torch.sum(attention_weights * region_feature, dim=1)
            else:
                # Method 2: Learn weights from concatenated features and attention scores
                combine_feature = torch.cat([region_feature, similarity_vector], dim=2)  # [B, N, region_dim + heads]
                attention_weights = self.fc_weight_learn(combine_feature)  # [B, N, 1]
                attention_weights = F.softmax(attention_weights, dim=1)
                agg_region_feature = torch.sum(attention_weights * region_feature, dim=1)
        
        # --------------------------------------------------------------------
        # STEP 6: Final classification
        # --------------------------------------------------------------------
        if opt.without_global_feature:
            # Use only region features (no global context)
            prediction = self.prediction_head(agg_region_feature)
        else:
            # Concatenate global and region features
            prediction = self.prediction_head(torch.cat([global_feature, agg_region_feature], dim=-1))
        
        return prediction, attention_weights


# ============================================================================
# TEST CODE
# ============================================================================
if __name__ == '__main__':
    """
    Quick verification script to test forward pass with random inputs.
    """
    b = 4  # Batch size
    img_cat = torch.randn(b, 4, 256, 256).cuda()
    target_box = torch.randint(size=(b, 4), low=0, high=256).float().cuda()
    refer_box = torch.randint(size=(b, 5, 6), low=0, high=256).float().cuda()  # 5 reference objects, 6 values per box
    target_feat = torch.randn(b, 1, 2048).cuda()
    refer_feat = torch.randn(b, 5, 2048).cuda()
    target_mask = torch.randn(b, 1, 64, 64).cuda()
    refer_mask = torch.randn(b, 5, 64, 64).cuda()
    w = h = (torch.ones(b) * 256).cuda()
    
    model = ObjectPlaceNet(backbone_pretrained=False).cuda()
    local_pre = model(img_cat, target_box, refer_box, target_feat, refer_feat, target_mask, refer_mask, w, h)
    print(local_pre)