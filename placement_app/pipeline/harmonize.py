"""
Image harmonization.  Tries PCTNet first (deep-learning ViT-based colour
transform), falls back to Reinhard (Lab-space statistics matching).

Reference:
    Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001.
    PCTNet: Pixel-Color-Transform Network (from BCMI/libcom).
"""

import os
import numpy as np
import cv2
from PIL import Image

# Lazy-loaded PCTNet instance
_pctnet = None


def _get_pctnet():
    global _pctnet
    if _pctnet is None:
        try:
            weight = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),
                'models', 'weights', 'PCTNet.pth')
            lut = os.path.join(os.path.dirname(weight), 'IdentityLUT33.txt')
            if os.path.exists(weight):
                from models.pct_net import PCTNetHarmonizer
                _pctnet = PCTNetHarmonizer(weight, lut, device='cuda')
        except Exception as e:
            print(f'[harmonize] PCTNet not available: {e}')
    return _pctnet


def harmonize(composite: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Harmonize foreground colours to match the background.

    Uses PCTNet if available; otherwise falls back to Reinhard.

    Args:
        composite: RGB composite image.
        mask:      L-mode mask, 255 = foreground.

    Returns:
        Harmonized RGB image.
    """
    pct = _get_pctnet()
    if pct is not None:
        try:
            return pct.harmonize(composite, mask)
        except Exception:
            pass
    return reinhard_harmonize(composite, mask)


def reinhard_harmonize(composite: Image.Image,
                       mask: Image.Image) -> Image.Image:
    """Reinhard colour transfer in CIE L*a*b* space."""
    comp_bgr = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
    mask_arr = np.array(mask)
    mask_bin = np.where(mask_arr > 127, 255, 0).astype(np.uint8)

    if mask_bin.shape[:2] != comp_bgr.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (comp_bgr.shape[1], comp_bgr.shape[0]))

    comp_lab = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2Lab)
    bg_mean, bg_std = cv2.meanStdDev(comp_lab, mask=255 - mask_bin)
    fg_mean, fg_std = cv2.meanStdDev(comp_lab, mask=mask_bin)
    fg_std_safe = np.where(fg_std < 1e-6, 1.0, fg_std)

    ratio = (bg_std / fg_std_safe).flatten()
    offset = (bg_mean - fg_mean * bg_std / fg_std_safe).flatten()

    trans_lab = cv2.convertScaleAbs(
        comp_lab.astype(np.float32) * ratio.reshape(1, 1, 3)
        + offset.reshape(1, 1, 3))
    trans_bgr = cv2.cvtColor(trans_lab, cv2.COLOR_Lab2BGR)

    mask_3ch = np.stack([mask_bin] * 3, axis=-1).astype(bool)
    result_bgr = comp_bgr.copy()
    result_bgr[mask_3ch] = trans_bgr[mask_3ch]

    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
