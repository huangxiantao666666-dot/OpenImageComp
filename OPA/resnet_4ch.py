"""
Script Name: resnet_4ch.py
Purpose: Modified ResNet architecture that accepts 4-channel input (RGB + Mask) for OPA task.

This script provides:
1. Standard ResNet building blocks (BasicBlock, Bottleneck)
2. A modified ResNet class that can accept variable input channels
3. Smart weight initialization for the extra channel using grayscale conversion formula
4. Support for both 4-channel (RGB+Mask) and 2-channel (depth) inputs

The key modification is replacing the first convolutional layer from Conv2d(3,64) 
to Conv2d(4,64) while preserving pretrained weights for RGB channels and 
intelligently initializing the mask channel.
"""

import torch
import torch.nn as nn
import math
from torchsummary import summary
import torch.utils.model_zoo as model_zoo
import os
from config import opt

# Pretrained model URLs from PyTorch official repository
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# ==================== BASIC CONVOLUTION BLOCKS ====================
def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding to preserve spatial dimensions.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels  
        stride: Convolution stride (1 or 2 for downsampling)
    
    Returns:
        Conv2d layer with kernel_size=3, padding=1, no bias
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34.
    
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add residual -> ReLU
    """
    expansion = 1  # Output channels = planes * expansion (no expansion for BasicBlock)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Args:
            inplanes: Input channel dimension
            planes: Output channel dimension (before expansion)
            stride: Stride for the first convolution (1 or 2)
            downsample: Downsample layer for shortcut connection when dimensions mismatch
        """
        super(BasicBlock, self).__init__()
        # First convolution (may downsample if stride=2)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Second convolution (always stride=1, preserves dimensions)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # Save input for skip connection

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsample to residual if needed (dimension mismatch)
        if self.downsample is not None:
            residual = self.downsample(x)

        # Skip connection (residual addition)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Residual Block for ResNet-50, ResNet-101, ResNet-152.
    
    Structure: Conv1x1 (reduce) -> Conv3x3 -> Conv1x1 (expand) -> BN -> Add residual -> ReLU
    
    This block reduces computation by first reducing channel dimension, then performing 3x3 conv,
    then expanding back to 4x the original dimension.
    """
    expansion = 4  # Output channels = planes * 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Args:
            inplanes: Input channel dimension
            planes: Intermediate channel dimension (output will be planes * 4)
            stride: Stride for the 3x3 convolution (1 or 2)
            downsample: Downsample layer for shortcut connection
        """
        super(Bottleneck, self).__init__()
        # 1x1 convolution: reduce dimension from inplanes to planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 convolution: spatial processing (may downsample)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 convolution: expand dimension from planes to planes*4
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # 1x1 conv (reduce)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv (spatial)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 conv (expand)
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ==================== MODIFIED RESNET ARCHITECTURE ====================
class ResNet(nn.Module):
    """
    Modified ResNet that can accept variable input channels.
    
    The standard ResNet expects 3-channel RGB input. This class allows
    modification of the first convolutional layer to accept different
    numbers of input channels.
    """

    def __init__(self, block, layers, num_classes=1000):
        """
        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: List of block counts for each stage [layer1, layer2, layer3, layer4]
            num_classes: Number of output classes (default 1000 for ImageNet)
        """
        self.inplanes = 64  # Initial number of channels after first convolution
        super(ResNet, self).__init__()
        
        # Initial layers (will be modified later for 4-channel input)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global average pooling to 1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # Fully connected layer

        # Weight initialization for all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization (Kaiming normal) for Conv2d
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BN weights to 1, bias to 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a sequential layer consisting of multiple residual blocks.
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            planes: Output channel dimension for this layer
            blocks: Number of blocks in this layer
            stride: Stride for the first block (1 or 2, determines downsampling)
        
        Returns:
            Sequential container of residual blocks
        """
        downsample = None
        
        # Create downsample layer if dimensions will change
        # Condition: stride != 1 OR input channels != output channels
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block (may have downsample and/or stride=2)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # Update input channels for next blocks
        
        # Remaining blocks (no downsampling, stride=1)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
               channels can be 3 (RGB) or 4 (RGB+Mask) or 2 (depth)
        
        Returns:
            Output logits of shape [batch, num_classes]
        """
        # Initial convolution + pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Shape: [batch, 64, H/4, W/4]

        # Residual stages
        x = self.layer1(x)  # [batch, 64*expansion, H/4, W/4]
        x = self.layer2(x)  # [batch, 128*expansion, H/8, W/8]
        x = self.layer3(x)  # [batch, 256*expansion, H/16, W/16]
        x = self.layer4(x)  # [batch, 512*expansion, H/32, W/32]

        # Classification head
        x = self.avgpool(x)  # [batch, 512*expansion, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten: [batch, 512*expansion]
        x = self.fc(x)  # [batch, num_classes]

        return x


# ==================== MODEL CONSTRUCTOR FUNCTIONS ====================
def resnet(layers, pretrained=False, pretrained_weight=None, **kwargs):
    """
    Create a ResNet model with optional 4-channel input support for OPA task.
    
    This function:
    1. Builds the standard ResNet architecture
    2. If opt.without_mask is False, replaces conv1 to accept 4 channels (RGB+Mask)
    3. If pretrained, intelligently initializes the extra mask channel using
       the grayscale conversion formula
    
    Args:
        layers: ResNet depth (18, 34, 50, 101, 152)
        pretrained: Whether to load pretrained weights
        pretrained_weight: Path to pretrained weight file
        **kwargs: Additional arguments passed to ResNet
    
    Returns:
        Modified ResNet model
    """
    # Step 1: Build standard ResNet based on layer depth
    model = None
    if layers == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif layers == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif layers == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif layers == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif layers == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    else:
        raise Exception('Unsupport resnet layers numbers: ', layers)
    
    # Step 2: Handle input channel modification based on mask usage
    if opt.without_mask:
        # Case 1: No mask input - use standard 3-channel RGB input
        if pretrained:
            model.load_state_dict(torch.load(pretrained_weight))
            print('loaded pretrained resnet{} from {}'.format(layers, pretrained_weight))
    else:
        # Case 2: With mask input - modify to accept 4 channels (RGB + Mask)
        
        # Replace first convolutional layer to accept 4 input channels
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            # Load pretrained weights
            pretrained_state_dict = torch.load(pretrained_weight)
            
            # Get the original 3-channel conv1 weights: shape [64, 3, 7, 7]
            conv1 = pretrained_state_dict['conv1.weight']
            
            # Create initial weights for the 4th (mask) channel: shape [64, 1, 7, 7]
            new = torch.zeros(64, 1, 7, 7)
            
            # Initialize mask channel using RGB-to-grayscale conversion formula
            # Formula: Gray = 0.299*R + 0.587*G + 0.114*B
            # This gives the mask channel a meaningful initialization rather than random
            for i, output_channel in enumerate(conv1):
                new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
            
            # Concatenate original 3-channel weights (RGB) with new mask channel weights
            # Result shape: [64, 4, 7, 7]
            pretrained_state_dict['conv1.weight'] = torch.cat((conv1, new), dim=1)
            
            # Load the modified state dict
            model.load_state_dict(pretrained_state_dict)
            print('loaded pretrained resnet{} from {}'.format(layers, pretrained_weight))
    
    return model


def resnet_for_depth(layers, pretrained=False, pretrained_weight=None, **kwargs):
    """
    Create a ResNet model for 2-channel input (e.g., depth maps).
    
    This function is similar to resnet() but modifies conv1 to accept 2 channels.
    Both channels are initialized using the grayscale conversion formula.
    
    Args:
        layers: ResNet depth (18, 34, 50, 101, 152)
        pretrained: Whether to load pretrained weights
        pretrained_weight: Path to pretrained weight file
        **kwargs: Additional arguments passed to ResNet
    
    Returns:
        Modified ResNet model for 2-channel input
    """
    # Step 1: Build standard ResNet
    model = None
    if layers == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif layers == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif layers == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif layers == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif layers == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    else:
        raise Exception('Unsupport resnet layers numbers: ', layers)

    # Step 2: Replace conv1 to accept 2 input channels
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    if pretrained:
        pretrained_state_dict = torch.load(pretrained_weight)
        conv1 = pretrained_state_dict['conv1.weight']  # [64, 3, 7, 7]
        
        # Create grayscale weights from RGB
        new = torch.zeros(64, 1, 7, 7)
        for i, output_channel in enumerate(conv1):
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        
        # For 2-channel input, both channels get the same grayscale initialization
        # Result shape: [64, 2, 7, 7]
        pretrained_state_dict['conv1.weight'] = torch.cat((new, new), dim=1)
        
        model.load_state_dict(pretrained_state_dict)
        print('loaded pretrained resnet{} from {}'.format(layers, pretrained_weight))
    
    return model


# ==================== TEST CODE ====================
if __name__ == '__main__':
    # Test the 4-channel ResNet with random input
    model = resnet(101, pretrained=False)
    input = torch.randn(2, 4, 256, 256)  # Batch=2, Channels=4, Height=256, Width=256
    print('Input shape {}, output shape {}'.format(input.shape, model(input).shape))