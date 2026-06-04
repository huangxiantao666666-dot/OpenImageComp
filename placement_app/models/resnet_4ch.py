"""
4-channel ResNet backbone for Object Placement Assessment.

Decoupled from the OPA project's global ``opt`` singleton.  All configuration
is passed as function / constructor arguments so that multiple model variants
(e.g. original vs. lightweight) can coexist in the same process.

Key modifications over standard torchvision ResNet:
1. First conv layer accepts 4 channels (RGB + mask) instead of 3.
2. ``base_width`` controls the stem channel count, enabling lightweight variants.
3. The 4th (mask) channel is initialised with the RGB→greyscale formula.
"""

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Pretrained model URLs (PyTorch official)
# ---------------------------------------------------------------------------
MODEL_URLS = {
    'resnet18':  'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':  'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':  'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# ======================================================================
#  Basic building blocks
# ======================================================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet-18 / 34 basic block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet-50 / 101 / 152 bottleneck block."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ======================================================================
#  ResNet (configurable base_width)
# ======================================================================

class ResNet(nn.Module):
    """ResNet whose channel widths are scaled by ``base_width``.

    Channel progression:  [w, 2w, 4w, 8w]
    where w = ``base_width`` (default 64 → classic ResNet).
    """

    def __init__(self, block, layers, num_classes=1000, base_width=64,
                 in_channels=3):
        super().__init__()
        self.base_width = base_width
        self.inplanes = base_width

        # Stem
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, base_width,      layers[0])
        self.layer2 = self._make_layer(block, base_width * 2,  layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4,  layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8,  layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # feature map, not logits (caller handles pool+fc)


# ======================================================================
#  Builder
# ======================================================================

BLOCK_TABLE = {
    18:  (BasicBlock, [2, 2, 2, 2]),
    34:  (BasicBlock, [3, 4, 6, 3]),
    50:  (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


def build_resnet(layers: int, base_width: int = 64,
                 pretrained: bool = True,
                 pretrained_weight: str = None,
                 in_channels: int = 4,
                 num_classes: int = 1000) -> ResNet:
    """
    Build a ResNet with configurable base_width and input channels.

    Args:
        layers:           ResNet depth (18, 34, 50, 101, 152).
        base_width:       Stem channel count (64 = classic, 32 = lightweight).
        pretrained:       Whether to load ImageNet pretrained weights.
        pretrained_weight:Path to a local .pth file (overrides auto-download).
        in_channels:      Number of input channels (4 = RGB + mask).
        num_classes:      Number of output classes.

    Returns:
        ResNet instance.
    """
    if layers not in BLOCK_TABLE:
        raise ValueError(f'Unsupported ResNet depth: {layers}')

    block, layer_cfg = BLOCK_TABLE[layers]
    model = ResNet(block, layer_cfg, num_classes=num_classes,
                   base_width=base_width, in_channels=in_channels)

    if not pretrained:
        return model

    # ---- load weights ---------------------------------------------------
    if pretrained_weight is None:
        pretrained_weight = MODEL_URLS.get(f'resnet{layers}')

    state_dict = torch.load(pretrained_weight, map_location='cpu')

    # Handle the 4th input channel (mask): initialise via greyscale formula
    if in_channels == 4:
        conv1_key = 'conv1.weight'
        if conv1_key in state_dict:
            old_w = state_dict[conv1_key]                     # [64, 3, 7, 7]
            new_ch = torch.zeros(old_w.size(0), 1, 7, 7)      # [64, 1, 7, 7]
            for i in range(old_w.size(0)):
                new_ch[i] = (0.299 * old_w[i, 0] +
                             0.587 * old_w[i, 1] +
                             0.114 * old_w[i, 2])
            state_dict[conv1_key] = torch.cat([old_w, new_ch], dim=1)

    # When base_width ≠ 64, only load layers whose shapes match
    if base_width != 64:
        model_state = model.state_dict()
        matched, skipped = 0, 0
        for k, v in list(state_dict.items()):
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                matched += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f'[ResNet] base_width={base_width}: {matched} layers loaded, '
              f'{skipped} skipped (shape mismatch)')
    else:
        # Remove fc if num_classes differs (we don't use fc anyway)
        if num_classes != 1000:
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
        model.load_state_dict(state_dict, strict=False)
        print(f'[ResNet] base_width={base_width}: full pretrained weights loaded')

    return model
