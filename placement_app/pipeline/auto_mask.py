"""
Auto-mask generation for foreground images without a provided mask.

Two backends, tried in order:
  1. SAM 2.1  (Meta, ``sam2.1_hiera_small.pt``) — centre-point prompt, best quality.
  2. OpenCV   (border-colour + Otsu)          — lightweight fallback.

Set ``SAM2_CHECKPOINT`` to the path of the downloaded checkpoint, or place it
at ``project/placement_app/models/weights/sam2.1_hiera_small.pt``.

If SAM 2 is not installed, the OpenCV backend is used silently.
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  SAM 2 backend
# ---------------------------------------------------------------------------
_sam2_predictor = None
_SAM2_LOADED = False
_SAM2_ERROR = None

# Look for the checkpoint in a few common locations
_SAM2_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'models', 'weights', 'sam2.1_hiera_small.pt'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
        'checkpoints', 'sam2.1_hiera_small.pt'),
]


def _load_sam2() -> bool:
    """Try to import and initialise SAM 2.1.  Returns True on success."""
    global _sam2_predictor, _SAM2_LOADED, _SAM2_ERROR
    if _SAM2_LOADED or _SAM2_ERROR is not None:
        return _SAM2_LOADED

    try:
        import torch
        # SAM 2.1 ships as the ``sam2`` package from Meta's GitHub repo
        # (pip install git+https://github.com/facebookresearch/sam2.git)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Find checkpoint
        ckpt = None
        for p in _SAM2_PATHS:
            if os.path.exists(p):
                ckpt = p
                break
        if ckpt is None:
            _SAM2_ERROR = 'sam2.1_hiera_small.pt not found'
            print(f'[auto_mask] SAM2 checkpoint not found. Tried: {_SAM2_PATHS}')
            return False

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',
                            ckpt, device=device)
        _sam2_predictor = SAM2ImagePredictor(model)
        _SAM2_LOADED = True
        print(f'[auto_mask] SAM2.1 loaded on {device}')
        return True

    except ImportError as e:
        _SAM2_ERROR = str(e)
        print(f'[auto_mask] SAM2 not installed ({e}), using OpenCV fallback')
        return False
    except Exception as e:
        _SAM2_ERROR = str(e)
        print(f'[auto_mask] SAM2 init failed: {e}')
        return False


def _sam2_mask(fg_rgb: np.ndarray) -> np.ndarray:
    """Generate mask with SAM 2.1 using a centre-point prompt."""
    if not _load_sam2():
        return _cv_mask(fg_rgb)

    h, w = fg_rgb.shape[:2]
    _sam2_predictor.set_image(fg_rgb)

    # Centre-point prompt with a bounding box covering most of the image
    cx, cy = w // 2, h // 2
    points = np.array([[cx, cy]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)  # 1 = foreground
    # A generous bounding box around the centre
    box = np.array([[w // 10, h // 10, w * 9 // 10, h * 9 // 10]],
                   dtype=np.float32)

    masks, scores, _ = _sam2_predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )

    # Pick the mask with the highest score
    best = scores.argmax()
    mask = masks[best].astype(np.uint8) * 255  # SAM2 returns bool, convert to 0/255
    fg_pct = (mask > 127).sum() / mask.size * 100
    print(f'[auto_mask] SAM2 mask: score={scores[best]:.3f}, fg={fg_pct:.1f}%')
    return mask


# ---------------------------------------------------------------------------
#  OpenCV backend (fallback)
# ---------------------------------------------------------------------------
def _cv_mask(fg_rgb: np.ndarray) -> np.ndarray:
    """Generate mask using border-colour estimation + Otsu thresholding."""
    h, w = fg_rgb.shape[:2]
    border_w = max(3, min(w, h) // 20)

    # Estimate background colour from border pixels
    borders = np.concatenate([
        fg_rgb[:border_w, :].reshape(-1, 3),
        fg_rgb[-border_w:, :].reshape(-1, 3),
        fg_rgb[:, :border_w].reshape(-1, 3),
        fg_rgb[:, -border_w:].reshape(-1, 3),
    ], axis=0)

    bg_colour = borders.mean(axis=0).astype(np.float32)

    # Colour distance
    diff = fg_rgb.astype(np.float32) - bg_colour.reshape(1, 1, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bin_mask = cv2.threshold(dist_norm, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest = sizes.argmax() + 1
        bin_mask = ((labels == largest).astype(np.uint8)) * 255

    # Morph-close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    return bin_mask


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def auto_mask(fg_rgb: np.ndarray, prefer_sam: bool = True) -> np.ndarray:
    """
    Generate a binary foreground mask.

    Tries SAM 2.1 first (if available and ``prefer_sam=True``), falls back
    to OpenCV border-colour + Otsu.

    Args:
        fg_rgb:     numpy array [H, W, 3] uint8.
        prefer_sam: try SAM2 before OpenCV.

    Returns:
        mask:  numpy array [H, W] uint8, 255 = foreground, 0 = background.
    """
    if prefer_sam and _load_sam2():
        return _sam2_mask(fg_rgb)
    return _cv_mask(fg_rgb)
