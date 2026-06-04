"""
PCTNet image harmonization wrapper.

Weights: ``PCTNet.pth`` (~19 MB) + ``IdentityLUT33.txt``
"""

import os
import numpy as np
import torch
from PIL import Image

from ._pct_net_lib import PCTNet


class PCTNetHarmonizer:
    """Loads PCTNet and harmonizes composite images.

    Args:
        weight_path:   Path to ``PCTNet.pth``.
        lut_path:      Path to ``IdentityLUT33.txt`` (bundled with libcom).
        image_size:    Working resolution for the low-res ViT branch.
        device:        'cuda' or 'cpu'.
    """

    def __init__(self, weight_path: str, lut_path: str = None,
                 image_size: int = 256, device: str = 'cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # Build model
        self.model = PCTNet(
            backbone_type='ViT',
            input_normalization={'mean': [0.0, 0.0, 0.0],
                                 'std': [1.0, 1.0, 1.0]},
            dim=3,
            transform_type='linear',
            affine=True,
            clamp=True,
            color_space='RGB',
            use_attn=False,
        ).eval().to(self.device)

        # Load weights
        state = torch.load(weight_path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

        # Load Identity LUT (required by PCT colour transform)
        if lut_path is None:
            lut_path = os.path.join(os.path.dirname(weight_path),
                                    'IdentityLUT33.txt')
        if os.path.exists(lut_path):
            LUT = np.loadtxt(lut_path).astype(np.float32)
        else:
            # Fallback: identity 3D LUT
            LUT = np.zeros((3, 33, 33, 33), dtype=np.float32)
            for c in range(3):
                LUT[c] = np.linspace(0, 1, 33).reshape(1, 1, 33)
        self._lut = torch.from_numpy(LUT).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'[PCTNet] Loaded {weight_path}  ({n_params/1e6:.1f}M params)')

    @torch.no_grad()
    def harmonize(self, composite: Image.Image,
                  mask: Image.Image) -> Image.Image:
        """
        Harmonize the foreground region of a composite image.

        Args:
            composite:  RGB composite image.
            mask:       L-mode mask, 255 = foreground.

        Returns:
            Harmonized RGB image (same size as input).
        """
        import torchvision.transforms as T

        orig_w, orig_h = composite.size

        # Low-res branch: 256x256
        to_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])

        comp_t = to_tensor(composite).to(self.device)     # [3, 256, 256]
        mask_t = to_tensor(mask).to(self.device)          # [1, 256, 256]

        # Full-res branch (PCTNet expects these without batch dim)
        to_tensor_fr = T.Compose([T.ToTensor()])
        fr_comp = to_tensor_fr(composite.resize(
            (orig_w, orig_h), Image.LANCZOS)).to(self.device)
        fr_mask = to_tensor_fr(mask.resize(
            (orig_w, orig_h), Image.LANCZOS)).to(self.device)

        # Forward (model does its own unsqueeze)
        output = self.model(comp_t, image_fullres=fr_comp,
                            mask=mask_t, mask_fullres=fr_mask)
        if isinstance(output, (list, tuple)):
            result = output[0]
        else:
            result = output

        result = result.squeeze(0).cpu().clamp(0, 1)
        result = (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(result)
