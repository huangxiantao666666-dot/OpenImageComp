"""
Grad-CAM visualisation for SimOPA model interpretation.

Generates class-activation heatmaps showing which image regions the model
attends to when judging placement rationality.

Uses tensor-level hooks (``register_hook``) rather than module-level backward
hooks to avoid the in-place residual-addition conflict in BasicBlock.

Usage:
    from grad_cam import GradCAM
    cam = GradCAM(scorer.model)
    heatmap = cam.generate(composite_tensor)   # numpy [256, 256]
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class GradCAM:
    """Grad-CAM for the last conv layer of SimOPA's backbone (layer4 output).

    Layers:
        model.backbone[-2]  → layer4 (nn.Sequential of 2 BasicBlocks)
        model.backbone[-1]  → pool (AdaptiveAvgPool2d)

    We hook the *output* of layer4, which is the deepest spatial feature map
    (8×8 for input 256×256).  The hook captures the activation tensor, then
    a tensor backward-hook captures its gradient.
    """

    def __init__(self, model):
        self.model = model
        self._activation = None
        self._gradient = None

        # Hook layer4 (model.backbone is nn.Sequential of [conv1,bn1,...,layer4])
        target = model.backbone[-1]

        def _forward_hook(module, inp, out):
            # Register a tensor hook on the output so we capture gradients
            def _tensor_hook(grad):
                self._gradient = grad.detach()
            out.register_hook(_tensor_hook)
            self._activation = out.detach()

        self._hook = target.register_forward_hook(_forward_hook)

    def remove(self):
        """Remove the registered hook (call when done)."""
        self._hook.remove()

    def generate(self, img_tensor: torch.Tensor,
                 class_idx: int = 1) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            img_tensor:  [1, 4, H, W]  on the correct device;
                         must have ``requires_grad=True``.
            class_idx:   0 = unreasonable, 1 = reasonable.

        Returns:
            numpy array [H, W]  values in [0, 1]; 1 = strongest activation.
        """
        self._activation = None
        self._gradient = None

        self.model.eval()

        # Ensure input tracks gradients
        img = img_tensor.clone()
        img.requires_grad_(True)

        # Forward
        logits = self.model(img)
        self.model.zero_grad()

        # Backward from target class
        score = logits[0, class_idx]
        score.backward()

        if self._activation is None or self._gradient is None:
            raise RuntimeError(
                'Grad-CAM hooks did not fire. Check that the model backbone '
                'structure matches: backbone[-1] should be layer4.')

        acts = self._activation                                   # [1, C, H, W]
        grads = self._gradient                                    # [1, C, H, W]

        # Channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)           # [1, C, 1, 1]

        # Weighted sum + ReLU
        cam = (weights * acts).sum(dim=1, keepdim=True)          # [1, 1, H, W]
        cam = torch.relu(cam)

        # Upsample
        _, _, h_in, w_in = img_tensor.shape
        cam = F.interpolate(cam, size=(h_in, w_in),
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        # Normalise
        denom = cam.max() - cam.min()
        cam = (cam - cam.min()) / max(denom, 1e-8)
        return cam

    def overlay(self, img_tensor: torch.Tensor, class_idx: int = 1,
                alpha: float = 0.5) -> Image.Image:
        """Generate a Grad-CAM heatmap + overlay on the RGB channels."""
        cam = self.generate(img_tensor, class_idx)

        # RGB channels (first 3 of the 4-channel input)
        rgb = img_tensor[0, :3].detach().cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-8)

        heatmap = _apply_jet(cam)
        blended = rgb * (1 - alpha) + heatmap * alpha
        blended = np.clip(blended, 0, 1)
        return Image.fromarray((blended * 255).astype(np.uint8))


# -----------------------------------------------------------------------
#  Jet colormap (no matplotlib dependency)
# -----------------------------------------------------------------------
def _apply_jet(grey: np.ndarray) -> np.ndarray:
    """Apply a jet colourmap to a [H, W] array in [0, 1]."""
    coloured = np.zeros((*grey.shape, 3), dtype=np.float32)
    coloured[..., 0] = np.clip(np.minimum(4 * grey - 1.5, -4 * grey + 4.5), 0, 1)
    coloured[..., 1] = np.clip(np.minimum(4 * grey - 0.5, -4 * grey + 3.5), 0, 1)
    coloured[..., 2] = np.clip(np.minimum(4 * grey + 0.5, -4 * grey + 2.5), 0, 1)
    return coloured
