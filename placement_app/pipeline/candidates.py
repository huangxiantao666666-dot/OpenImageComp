"""
Generate candidate bounding boxes for object placement.
"""

from typing import List, Dict


def generate_candidates(bg_width: int, bg_height: int,
                        fg_width: int, fg_height: int,
                        grid_size: int = 5,
                        n_scales: int = 3) -> List[Dict]:
    """
    Place a regular grid of candidate locations across the background.

    Args:
        bg_width, bg_height:  Background dimensions (pixels).
        fg_width, fg_height:  Original foreground dimensions (pixels).
        grid_size:            Number of grid points per axis (5 → 5×5 = 25).
        n_scales:             How many scales to try (1→3 scales centred on 1.0).

    Returns:
        List of dicts, each with:
            ``bbox``  : [x1, y1, x2, y2] in background coordinates.
            ``scale`` : scale factor applied (e.g. 0.8 / 1.0 / 1.2).
    """
    # Scale factors — adapt range to foreground/background size ratio
    # If fg is larger than bg, we need smaller scales.
    max_ratio_w = bg_width / max(fg_width, 1)
    max_ratio_h = bg_height / max(fg_height, 1)
    max_possible = min(max_ratio_w, max_ratio_h, 1.2)

    if n_scales == 1:
        scales = [min(1.0, max_possible * 0.9)]
    elif n_scales == 2:
        scales = [max(0.3, max_possible * 0.5), max_possible * 0.9]
    else:
        min_s = max(0.2, max_possible * 0.3)
        step = (max_possible - min_s) / (n_scales - 1)
        scales = [round(min_s + i * step, 2) for i in range(n_scales)]

    candidates = []
    for scale in scales:
        w = int(fg_width * scale)
        h = int(fg_height * scale)

        # Skip if scaled foreground is larger than background
        if w >= bg_width or h >= bg_height or w <= 0 or h <= 0:
            continue

        x_step = max((bg_width - w) // (grid_size + 1), 1)
        y_step = max((bg_height - h) // (grid_size + 1), 1)

        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x1 = x_step * i
                y1 = y_step * j
                candidates.append({
                    'bbox':  [x1, y1, x1 + w, y1 + h],
                    'scale': scale,
                })

    return candidates


def generate_around_point(cx: int, cy: int, fg_w: int, fg_h: int,
                           bg_w: int, bg_h: int,
                           radius: int = 40, n_samples: int = 15) -> List[Dict]:
    """
    Generate candidates in a local neighbourhood around a user-clicked point.

    Args:
        cx, cy:   Centre point (pixels).
        fg_w, fg_h: Foreground size.
        bg_w, bg_h: Background size.
        radius:   Search radius in pixels.
        n_samples:Number of random jitter samples.

    Returns:
        List of candidate dicts (same format as ``generate_candidates``).
    """
    import random
    candidates = []
    for _ in range(n_samples):
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        x1 = max(0, min(cx + dx, bg_w - fg_w))
        y1 = max(0, min(cy + dy, bg_h - fg_h))
        candidates.append({
            'bbox':  [x1, y1, x1 + fg_w, y1 + fg_h],
            'scale': 1.0,
        })
    return candidates
