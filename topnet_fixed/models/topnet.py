"""
TopNet — Transformer-based Object Placement Network (CVPR 2023).

**FIXED VERSION**: The original implementation contains two bugs in the
Transformer block (see docstring of ``_TransformerModule`` for details).
This file provides a corrected version that should be trained from scratch
(on the official SOPA/OPA dataset) rather than loaded from the pretrained
checkpoint.

Architecture:
  Dual 4ch ResNet18 encoders → 4-layer Transformer → UNet decoder → [2,256,256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  BasicConv2d  (Conv + BN + ReLU)
# ======================================================================
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


# ======================================================================
#  ResNet backbone (minimal — adapted from our working placement_app copy)
# ======================================================================
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class _ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
        return x


# ======================================================================
#  Encoder builders
# ======================================================================
def _build_bg_encoder():
    """
    Background encoder — 4ch ResNet18 (SOPA-pretrained 4th channel).

    Split into 5 sub-modules for the UNet skip connections:
      div_2  (128²), div_4 (64²), div_8 (32²), div_16 (16²), div_32 (8²)
    """
    model = _ResNet(_BasicBlock, [2, 2, 2, 2])
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    div_2  = nn.Sequential(*list(model.children())[:3])     # conv1, bn1, relu
    div_4  = nn.Sequential(*list(model.children())[3:5])    # maxpool, layer1
    div_8  = model.layer2
    div_16 = model.layer3
    div_32 = model.layer4
    return div_2, div_4, div_8, div_16, div_32


def _build_fg_encoder():
    """
    Foreground encoder — 4ch ResNet18 (ImageNet pretrained + greyscale-init
    4th channel).

    Split into 6 sub-modules:
      div_1 (256²), div_2 (128²), div_4 (64²), div_8 (32²),
      div_16 (16²), div_32 (8²)
    """
    model = _ResNet(_BasicBlock, [2, 2, 2, 2])
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    div_1  = nn.Sequential(*list(model.children())[:1])     # conv1
    div_2  = nn.Sequential(*list(model.children())[1:3])    # bn1, relu
    div_4  = nn.Sequential(*list(model.children())[3:5])    # maxpool, layer1
    div_8  = model.layer2
    div_16 = model.layer3
    div_32 = model.layer4
    return div_1, div_2, div_4, div_8, div_16, div_32


# ======================================================================
#  Transformer  — **FIXED VERSION**
# ======================================================================
#
#  Original bugs:
#   1. ``LayerNorm(8)`` normalised across spatial columns (W=8) instead of
#      the feature dimension (1024).  Now uses ``LayerNorm(1024)``.
#   2. MHA input was ``[B, 64, 1024]`` with ``batch_first=False``, causing
#      the MHA to treat B (batch) as the sequence length.  At inference
#      (B=1) this rendered the attention degenerative (single-token identity).
#      Now uses ``batch_first=True`` so MHA properly attends across the
#      64 spatial tokens.
#   3. The huge MLP ``Linear(65536→128→65536)`` is kept because reducing it
#      would change the pretrained weight shapes.  If you train from scratch
#      you may want to replace it with a per-token MLP for efficiency.

class _TransformerModule(nn.Module):
    """Fixed: correct LayerNorm dim + batch_first MHA."""

    def __init__(self, embed_dim=1024, num_heads=8, mlp_expansion=4):
        """
        Args:
            embed_dim:     Feature dimension (1024).
            num_heads:     Attention heads (8).
            mlp_expansion: MLP hidden = embed_dim * mlp_expansion.
                           4 = 79M model, 8 = 113M model.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden = embed_dim * mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1024, 8, 8]

        Returns:
            [B, 1024, 8, 8]
        """
        B, C, H, W = x.shape

        # Reshape to tokens: [B, C, H, W] → [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2)                # [B, 64, 1024]

        # Self-attention with pre-norm
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)   # [B, 64, 1024]
        tokens = tokens + attn_out

        # MLP with pre-norm
        tokens = tokens + self.mlp(self.norm2(tokens))        # [B, 64, 1024]

        # Reshape back: [B, H*W, C] → [B, C, H, W]
        x = tokens.transpose(1, 2).view(B, C, H, W)
        return x


class _Transformer(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, num_layers=4,
                 mlp_expansion=4):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerModule(embed_dim, num_heads, mlp_expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ======================================================================
#  Decoder helpers
# ======================================================================
def _upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.shape[2:], mode='bilinear',
                              align_corners=False)
    return y


# ======================================================================
#  ObPlaNet_resnet18  (full model)
# ======================================================================
class ObPlaNet_resnet18(nn.Module):
    """
    Fixed TopNet discriminator for object placement.

    Total params: ~113M (original) / reduced with per-token MLP.
    """

    def __init__(self, out_channels: int = 2, mlp_expansion: int = 4):
        """
        Args:
            out_channels:  2 = binary classifier, 1 = keypoint heatmap.
            mlp_expansion: 4 = 79M model, 8 = 113M model.
        """
        super().__init__()
        self.out_channels = out_channels

        # ---- Encoders ----
        (self.bg_encoder1, self.bg_encoder2, self.bg_encoder4,
         self.bg_encoder8, self.bg_encoder16) = _build_bg_encoder()

        (self.fg_encoder1, self.fg_encoder2, self.fg_encoder4,
         self.fg_encoder8, self.fg_encoder16,
         self.fg_encoder32) = _build_fg_encoder()

        # ---- Transformer (FIXED) ----
        self.transformer = _Transformer(
            embed_dim=1024, num_heads=8, num_layers=4,
            mlp_expansion=mlp_expansion)

        # ---- UNet Decoder ----
        self.upconv32 = BasicConv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv8  = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4  = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2  = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1  = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv = nn.ConvTranspose2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(512, out_channels, 1)
        self.sigmoid = nn.Sigmoid() if out_channels == 1 else None

    def forward(self, bg, fg, mask):
        """
        Args:
            bg:   [B, 3, 256, 256]  background
            fg:   [B, 3, 256, 256]  foreground
            mask: [B, 1, 256, 256]  foreground mask

        Returns:
            If out_channels==2: logits [B, 2, 256, 256] (binary classifier)
            If out_channels==1: probs  [B, 1, 256, 256] (heatmap, 0=bad 1=good)
        """
        # Build 4ch inputs
        black = torch.zeros_like(mask)
        bg_4ch = torch.cat([bg, black], dim=1)               # [B, 4, 256²]
        fg_4ch = torch.cat([fg, mask], dim=1)               # [B, 4, 256²]

        # Encoders
        bg1 = self.bg_encoder1(bg_4ch)                       # [B, 64, 128²]
        fg1 = self.fg_encoder1(fg_4ch)
        bg2 = self.bg_encoder2(bg1)                          # [B, 64, 64²]
        fg2 = self.fg_encoder2(fg1)
        bg4 = self.bg_encoder4(bg2)                          # [B, 128, 32²]
        fg4 = self.fg_encoder4(fg2)
        bg8 = self.bg_encoder8(bg4)                          # [B, 256, 16²]
        fg8 = self.fg_encoder8(fg4)
        bg16 = self.bg_encoder16(bg8)                        # [B, 512, 8²]
        fg16 = self.fg_encoder16(fg8)
        fg32 = self.fg_encoder32(fg16)                       # [B, 512, 8²]

        # Transformer on concatenated deepest features
        x = torch.cat([bg16, fg32], dim=1)                  # [B, 1024, 8²]
        x = self.transformer(x)                              # [B, 1024, 8²]

        # UNet Decoder with skip connections
        x = self.upconv32(x)                                 # [B, 512, 8²]
        x = _upsample_add(self.upconv16(x), bg8)            # [B, 256, 16²]
        x = _upsample_add(self.upconv8(x), bg4)             # [B, 128, 32²]
        x = _upsample_add(self.upconv4(x), bg2)             # [B, 64, 64²]
        x = _upsample_add(self.upconv2(x), bg1)             # [B, 64, 128²]
        x = self.upconv1(F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=True))  # [B, 64, 256²]
        x = self.deconv(x)                                   # [B, 512, 256²]
        out = self.classifier(x)                             # [B, out_C, 256²]
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        return out


# ======================================================================
#  Test
# ======================================================================
class ObPlaNet_resnet18_keypoint(ObPlaNet_resnet18):
    """Keypoint-detection variant: output [B, 1, H, W] heatmap with Sigmoid."""

    def __init__(self, mlp_expansion: int = 4):
        super().__init__(out_channels=1, mlp_expansion=mlp_expansion)


# ======================================================================
#  Model factory
# ======================================================================
_MODEL_REGISTRY = {
    'ObPlaNet_resnet18': ObPlaNet_resnet18,
    'ObPlaNet_resnet18_keypoint': ObPlaNet_resnet18_keypoint,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a TopNet variant by name.

    Args:
        name:   'ObPlaNet_resnet18' (2ch),
                'ObPlaNet_resnet18_keypoint' (1ch).
                Append '_113M' suffix for mlp_expansion=8 version.
        kwargs: passed to the model constructor.

    Returns:
        nn.Module instance.
    """
    if name.endswith('_113M'):
        base = name.replace('_113M', '')
        kwargs.setdefault('mlp_expansion', 8)
        name = base

    if name not in _MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}. '
                         f'Available: {list(_MODEL_REGISTRY.keys())}')
    return _MODEL_REGISTRY[name](**kwargs)


# ======================================================================
#  Test
# ======================================================================
if __name__ == '__main__':
    m = ObPlaNet_resnet18()
    n = sum(p.numel() for p in m.parameters())
    print(f'Total params: {n:,}')

    bg = torch.randn(2, 3, 256, 256)
    fg = torch.randn(2, 3, 256, 256)
    mk = torch.randn(2, 1, 256, 256)
    out = m(bg, fg, mk)
    print(f'Input:  bg={bg.shape}, fg={fg.shape}, mask={mk.shape}')
    print(f'Output (2ch): {out.shape}')

    mkp = ObPlaNet_resnet18_keypoint()
    out_kp = mkp(bg, fg, mk)
    print(f'Output (1ch): {out_kp.shape}  range=[{out_kp.min():.4f}, {out_kp.max():.4f}]')
