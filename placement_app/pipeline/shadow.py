"""
Simple shadow rendering — no deep learning required.

Adds a soft, semi-transparent shadow below a placed foreground object.
The shadow is rendered as a blurred ellipse at the base of the object,
with size and opacity proportional to the object's bounding box.
"""

import numpy as np
import cv2
from PIL import Image


def add_shadow(composite: Image.Image,
               mask: Image.Image,
               bbox: list,
               opacity: float = 0.35,
               angle_deg: float = 0.0) -> Image.Image:
    """
    Render a soft drop shadow under a foreground object.

    Args:
        composite:  RGB composite image with foreground already placed.
        mask:       Foreground mask (L mode, 255 = foreground).
        bbox:       [x1, y1, x2, y2] of the placed foreground.
        opacity:    Shadow darkness (0 = invisible, 1 = fully opaque black).
        angle_deg:  Not used for simple shadow; reserved.

    Returns:
        RGB composite image with shadow added.
    """
    comp_arr = np.array(composite).astype(np.float32)
    h, w = comp_arr.shape[:2]

    x1, y1, x2, y2 = bbox

    # Shadow centre at the bottom of the object, slightly to the right
    cx = (x1 + x2) // 2 + int((x2 - x1) * 0.1)
    cy = y2 + int((y2 - y1) * 0.02)

    # Ellipse dimensions proportional to object size
    obj_w = x2 - x1
    obj_h = y2 - y1
    shadow_w = int(obj_w * 0.8)
    shadow_h = max(int(obj_h * 0.12), 4)

    # Create shadow on a separate layer
    shadow_layer = np.zeros((h, w), dtype=np.float32)

    # Draw filled ellipse
    cv2.ellipse(shadow_layer,
                (cx, cy),                        # centre
                (shadow_w // 2, shadow_h // 2),  # axes
                0, 0, 360,                       # angle, start, end
                1.0, -1)                         # colour, fill

    # Gaussian blur for soft edges
    blur_sigma = max(shadow_h / 3, 2.0)
    shadow_layer = cv2.GaussianBlur(shadow_layer, (0, 0), blur_sigma)

    # Apply shadow: darken background pixels
    shadow_3ch = np.stack([shadow_layer] * 3, axis=-1) * opacity
    comp_arr = comp_arr * (1 - shadow_3ch)

    return Image.fromarray(np.clip(comp_arr, 0, 255).astype(np.uint8))
