"""
SimOPA: Object Placement Assessment model.

Decoupled from the OPA project's global ``opt`` singleton.
Provides two interfaces:

- ``SimOPA`` (nn.Module):  backbone → pool → fc → logits
- ``SimOPAScorer``:         loads weights, exposes ``score(composite, mask) → float``
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from .resnet_4ch import build_resnet


# ======================================================================
#  SimOPA  model
# ======================================================================

class SimOPA(nn.Module):
    """Object placement rationality classifier.

    Args:
        backbone:         ResNet variant (``'resnet18'`` / ``'resnet34'`` / …).
        base_width:       Stem width; 64 = classic ResNet, 32 = lightweight.
        num_classes:      Number of output classes (2 = reasonable / unreasonable).
        pretrained:       Load ImageNet pretrained weights for the backbone.
        pretrained_weight:Path to backbone .pth file.
    """

    def __init__(self, backbone: str = 'resnet18', base_width: int = 64,
                 num_classes: int = 2, pretrained: bool = True,
                 pretrained_weight: str = None):
        super().__init__()

        resnet_layers = int(backbone.replace('resnet', ''))
        full_backbone = build_resnet(
            layers=resnet_layers,
            base_width=base_width,
            pretrained=pretrained,
            pretrained_weight=pretrained_weight,
            in_channels=4,
            num_classes=1000,          # not used – we'll drop fc
        )

        # Drop avgpool & fc – keep only conv feature extractor
        children = list(full_backbone.children())
        self.backbone = nn.Sequential(*children[:-2])

        # Feature dimension after backbone
        if base_width == 64:
            self.feat_dim = 512 if backbone in ('resnet18', 'resnet34') else 2048
        else:
            expansion = 4 if backbone not in ('resnet18', 'resnet34') else 1
            self.feat_dim = base_width * 8 * expansion

        self.pool = nn.AdaptiveAvgPool2d(1)
        # Name must match the original checkpoint key ('prediction_head')
        self.prediction_head = nn.Linear(self.feat_dim, num_classes, bias=False)

    def forward(self, img_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_cat:  [B, 4, H, W]  (RGB 3ch + mask 1ch concatenated)

        Returns:
            logits:  [B, num_classes]
        """
        feat = self.backbone(img_cat)       # [B, feat_dim, H/32, W/32]
        feat = self.pool(feat)              # [B, feat_dim, 1, 1]
        feat = feat.flatten(1)              # [B, feat_dim]
        return self.prediction_head(feat)   # [B, num_classes]


# ======================================================================
#  Scorer  (inference wrapper)
# ======================================================================

class SimOPAScorer:
    """Loads a trained SimOPA checkpoint and provides a simple ``score()`` API.

    Args:
        weight_path:  Path to a ``.pth`` checkpoint (e.g. ``simopa.pth``).
        backbone:     Backbone name, must match the saved checkpoint.
        base_width:   Stem width, must match the saved checkpoint.
        device:       ``'cuda'`` / ``'cpu'``.
    """

    def __init__(self, weight_path: str, backbone: str = 'resnet18',
                 base_width: int = 64, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = 256

        self.model = SimOPA(backbone=backbone, base_width=base_width,
                            pretrained=False)
        state = torch.load(weight_path, map_location='cpu')

        # The checkpoint may contain keys prefixed with 'module.' (DataParallel)
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f'[SimOPAScorer] Missing keys: {missing}')
        if unexpected:
            print(f'[SimOPAScorer] Unexpected keys: {unexpected}')

        self.model = self.model.eval().to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'[SimOPAScorer] Loaded {weight_path}')
        print(f'               backbone={backbone}, base_width={base_width}')
        print(f'               params={n_params:,}, device={self.device}')

        self._transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def score(self, composite: Image.Image, mask: Image.Image = None,
              mode: str = '4ch', bbox: list = None) -> float:
        """
        Score a composite image + foreground mask.

        Args:
            composite:  RGB composite image.
            mask:       Greyscale mask, 255 = foreground (needed for 4ch/crop modes).
            mode:       Input variant:
                '4ch'   — RGB + mask concatenated (default, original).
                '3ch'   — RGB only, mask ignored.
                'crop'  — Crop the region around ``bbox``, then score.
                          Uses 4ch if ``mask`` is given, else 3ch.
            bbox:       [x1, y1, x2, y2] for crop mode.

        Returns:
            Rationality score in [0, 1].  Higher → more reasonable placement.
        """
        if mode == 'crop' and bbox is not None:
            # Expand crop region by sqrt(2) for context
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            add_w, add_h = int(w * 0.207), int(h * 0.207)
            x1 = max(0, x1 - add_w)
            y1 = max(0, y1 - add_h)
            x2 = min(composite.width, x2 + add_w)
            y2 = min(composite.height, y2 + add_h)
            composite = composite.crop((x1, y1, x2, y2))
            if mask is not None:
                mask = mask.crop((x1, y1, x2, y2))

        img_t = self._transform(composite.convert('RGB'))          # [3, 256, 256]

        if mode == '3ch' or mask is None:
            cat = img_t.unsqueeze(0).to(self.device)              # [1, 3, 256, 256]
        else:
            msk_t = self._transform(mask.convert('L'))            # [1, 256, 256]
            cat = torch.cat([img_t, msk_t], dim=0).unsqueeze(0)   # [1, 4, 256, 256]
        cat = cat.to(self.device)

        if cat.shape[1] == 3:
            logits = self._forward_3ch(cat)
        else:
            logits = self.model(cat)

        prob = F.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)

    @torch.no_grad()
    def _forward_3ch(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass with 3-channel input, reusing 4ch conv1 weights."""
        # conv1: use first 3 channels of the 4ch weight
        conv1_w = self.model.backbone[0].weight  # [64, 4, 7, 7]
        bn1 = self.model.backbone[1]
        relu = self.model.backbone[2]
        maxpool = self.model.backbone[3]

        x = F.conv2d(img, conv1_w[:, :3], stride=2, padding=3)  # use 3ch only
        x = relu(bn1(x))
        x = maxpool(x)
        # Rest of backbone
        for layer in self.model.backbone[4:]:
            x = layer(x)
        feat = self.model.pool(x).flatten(1)
        return self.model.prediction_head(feat)
