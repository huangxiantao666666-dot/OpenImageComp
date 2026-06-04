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
    def score(self, composite: Image.Image, mask: Image.Image) -> float:
        """
        Score a composite image + foreground mask.

        Args:
            composite:  RGB composite image (any size, will be resized to 256²).
            mask:       Greyscale mask (any size), 255 = foreground region.

        Returns:
            Rationality score in [0, 1].  Higher → more reasonable placement.
        """
        img_t = self._transform(composite.convert('RGB'))          # [3, 256, 256]
        msk_t = self._transform(mask.convert('L'))                 # [1, 256, 256]
        cat   = torch.cat([img_t, msk_t], dim=0).unsqueeze(0)     # [1, 4, 256, 256]
        cat   = cat.to(self.device)

        logits = self.model(cat)                                   # [1, 2]
        prob   = F.softmax(logits, dim=-1)[0, 1].item()           # P(reasonable)
        return float(prob)
