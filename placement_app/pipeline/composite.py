"""
Composite image generation: paste a foreground onto a background at a given bbox.
"""

import numpy as np
from PIL import Image


def make_composite(bg: Image.Image, fg: Image.Image,
                   fg_mask: Image.Image, bbox: list) -> tuple:
    """
    Composite foreground onto background at the specified bounding box.

    Args:
        bg:       Background PIL image (RGB).
        fg:       Foreground PIL image (RGB), possibly with junk background.
        fg_mask:  Foreground mask (L mode), 255 = keep, 0 = discard.
        bbox:     [x1, y1, x2, y2] in background pixel coordinates.

    Returns:
        ``(composite, composite_mask)`` where both are PIL images at the
        background's native resolution:
            * ``composite`` — RGB, foreground blended with background.
            * ``composite_mask`` — L, 255 at foreground pixels, 0 elsewhere.
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Clip bbox to background boundaries
    x1 = max(0, bbox[0])
    y1 = max(0, bbox[1])
    x2 = min(bg.width, bbox[2])
    y2 = min(bg.height, bbox[3])
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        raise ValueError(f'Invalid bbox {bbox}: clipped to zero area')

    # Resize foreground + mask to the target bbox size
    fg_r = fg.resize((w, h), Image.LANCZOS)
    mk_r = fg_mask.resize((w, h), Image.LANCZOS)

    bg_arr = np.array(bg.copy()).astype(np.float32)
    fg_arr = np.array(fg_r).astype(np.float32)
    mk_arr = np.array(mk_r).astype(np.float32) / 255.0       # [0, 1]

    if mk_arr.ndim == 2:
        mk_arr = np.stack([mk_arr] * 3, axis=-1)             # → [H, W, 3]

    # Alpha blending:  composite = fg * α + bg * (1-α)
    bg_roi = bg_arr[y1:y2, x1:x2]
    bg_arr[y1:y2, x1:x2] = fg_arr * mk_arr + bg_roi * (1 - mk_arr)

    # Full-resolution mask (255 inside the bbox where mask was > 127)
    full_mask = np.zeros((bg.height, bg.width), dtype=np.uint8)
    roi_mask = (np.array(mk_r) > 127).astype(np.uint8) * 255
    full_mask[y1:y2, x1:x2] = roi_mask

    composite_img = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8))
    mask_img = Image.fromarray(full_mask, mode='L')

    return composite_img, mask_img
