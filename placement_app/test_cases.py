"""
Test case runner for Object Placement Assistant.

Tests the SimOPA model on pre-defined placement scenarios covering:
  - Reasonable placements (high score expected)
  - Unreasonable placements (low score expected)
  - Borderline cases (medium score expected)

Also runs model comparison if a second model is provided (e.g. lightweight vs full).
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SimOPAScorer
from pipeline import PlacementScorer, generate_candidates, make_composite
from grad_cam import GradCAM


# ---------------------------------------------------------------------------
#  Utility: create synthetic test images
# ---------------------------------------------------------------------------

def _create_synthetic_bg(size=(640, 480), sky=True):
    """Create a simple synthetic background: sky gradient + green ground."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if sky:
        # Blue sky gradient
        for y in range(size[1] * 2 // 3):
            arr[y, :, 2] = int(180 - y * 0.3)
            arr[y, :, 1] = int(200 - y * 0.2)
            arr[y, :, 0] = int(255 - y * 0.3)
    # Green ground
    arr[size[1] * 2 // 3:, :, 1] = 120
    arr[size[1] * 2 // 3:, :, 2] = 50
    return Image.fromarray(arr)


def _create_synthetic_fg(size=(120, 100), color=(200, 100, 50)):
    """Create a simple coloured rectangle with a circular mask."""
    img = Image.new('RGB', size, color)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([4, 4, size[0] - 4, size[1] - 4], fill=255)
    return img, mask


# ---------------------------------------------------------------------------
#  Pre-defined test cases
# ---------------------------------------------------------------------------

# Each case: (name, bg_creator, fg_creator, good_bbox, bad_bbox, expected)
# good_bbox / bad_bbox: [x1, y1, x2, y2] or None (auto)
TEST_CASES = []

# Scene 1: Sky + ground background, orange circle foreground
bg1 = _create_synthetic_bg()
fg1, mk1 = _create_synthetic_fg((100, 80), (220, 120, 40))

TEST_CASES.append({
    'name':       'Reasonable - foreground on ground',
    'bg':         bg1,
    'fg':         fg1,
    'fg_mask':    mk1,
    'good_bbox':  [200, 400, 300, 480],    # on ground
    'bad_bbox':   [100, 50, 200, 130],     # floating in sky
    'expected':   'good > bad by ≥ 0.3',
})

TEST_CASES.append({
    'name':       'Borderline - straddles horizon',
    'bg':         bg1,
    'fg':         fg1,
    'fg_mask':    mk1,
    'good_bbox':  [220, 330, 320, 410],    # straddles horizon
    'bad_bbox':   [10, 10, 110, 90],       # top-left corner
    'expected':   'horizon case: moderate score',
})

TEST_CASES.append({
    'name':       'Out-of-bounds - foreground exceeds bg',
    'bg':         bg1,
    'fg':         fg1,
    'fg_mask':    mk1,
    'good_bbox':  [50, 350, 150, 430],     # fully inside
    'bad_bbox':   [580, 420, 680, 500],     # right edge out
    'expected':   'out-of-bounds gets low score',
})

TEST_CASES.append({
    'name':       'Scale - oversized foreground',
    'bg':         bg1,
    'fg':         fg1,
    'fg_mask':    mk1,
    'good_bbox':  [260, 440, 360, 520],    # normal size
    'bad_bbox':   [200, 350, 300, 430],    # same position, wrong aspect
    'expected':   'reasonable scale > distorted scale',
})

# Scene 2: Use OPA example images
ex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'OPA', 'eval_opascore', 'examples')
if os.path.exists(ex_dir):
    real_bg = Image.open(os.path.join(ex_dir, 'composite_1.jpg')).convert('RGB')
    real_mask = Image.open(os.path.join(ex_dir, 'mask_1.jpg')).convert('L')
    # Extract foreground from composite using mask
    bg_arr = np.array(real_bg).astype(np.float32)
    mk_arr = np.array(real_mask).astype(np.float32) / 255.0
    mk_3ch = np.stack([mk_arr] * 3, axis=-1)
    # Foreground = composite * mask (rough extraction)
    fg_arr = (bg_arr * mk_3ch).astype(np.uint8)
    # Find bbox of mask
    ys, xs = np.where(mk_arr > 0.5)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    real_fg = Image.fromarray(fg_arr[y1:y2, x1:x2])
    real_fg_mask = Image.fromarray((mk_arr[y1:y2, x1:x2] * 255).astype(np.uint8), mode='L')

    TEST_CASES.append({
        'name':       'Real image - OPA example',
        'bg':         real_bg,
        'fg':         real_fg,
        'fg_mask':    real_fg_mask,
        'good_bbox':  [x1, y1, x2, y2],          # original placement
        'bad_bbox':   [max(0, x1-200), max(0, y1-100),
                       max(0, x1-200)+(x2-x1), max(0, y1-100)+(y2-y1)],
        'expected':   'original > shifted',
    })


# ---------------------------------------------------------------------------
#  Test runner
# ---------------------------------------------------------------------------

def run_tests(scorer_full: PlacementScorer,
              scorer_lite: PlacementScorer = None,
              save_dir: str = './outputs'):
    """
    Run all test cases and produce a comparison report.

    Args:
        scorer_full:  PlacementScorer wrapping the full SimOPA model.
        scorer_lite:  (Optional) lightweight model scorer for comparison.
        save_dir:     Where to save composite images and results.

    Returns:
        list of result dicts.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for i, case in enumerate(TEST_CASES):
        print(f'\n{"="*60}')
        print(f'  Case {i+1}: {case["name"]}')
        print(f'{"="*60}')

        bg, fg, fg_mask = case['bg'], case['fg'], case['fg_mask']

        scores = {}
        composites = {}

        for label, bbox in [('good', case['good_bbox']),
                            ('bad', case['bad_bbox'])]:
            if bbox is None:
                continue

            # Full model
            comp, mask = make_composite(bg, fg, fg_mask, bbox)
            t0 = time.time()
            score_full = scorer_full.model.score(comp, mask)
            t_full = time.time() - t0
            composites[label] = comp

            entry = {
                'case': case['name'],
                'label': label,
                'bbox': bbox,
                'score_full': score_full,
                'time_full_ms': t_full * 1000,
            }

            if scorer_lite:
                t0 = time.time()
                score_lite = scorer_lite.model.score(comp, mask)
                t_lite = time.time() - t0
                entry['score_lite'] = score_lite
                entry['time_lite_ms'] = t_lite * 1000

            scores[label] = entry
            results.append(entry)

            delta = (f'Δ={score_full:.4f}'
                     if not scorer_lite
                     else f'Δ={score_full:.4f} (lite={score_lite:.4f})')
            print(f'  {label:6s} {bbox} -> {delta}')

            # Save composite
            comp.save(os.path.join(save_dir,
                      f'case{i+1}_{label}_{case["name"]}.png'))

        # Summary for this case
        if 'good' in scores and 'bad' in scores:
            gap_full = scores['good']['score_full'] - scores['bad']['score_full']
            print(f'  -> Good - Bad (full) = {gap_full:+.4f}')
            if scorer_lite:
                gap_lite = (scores['good']['score_lite'] -
                            scores['bad']['score_lite'])
                print(f'  -> Good - Bad (lite) = {gap_lite:+.4f}')

    # ---- Summary table ----
    print(f'\n{"="*70}')
    print(f'  SUMMARY')
    print(f'{"="*70}')
    header = f'  {"Case":<32s} {"Good":>8s} {"Bad":>8s} {"Gap":>8s}'
    if scorer_lite:
        header += f' {"Good(L)":>8s} {"Bad(L)":>8s} {"Gap(L)":>8s} {"Time":>8s}'
    else:
        header += f' {"Time":>8s}'
    print(header)
    print(f'  {"-"*70}')

    for i, case in enumerate(TEST_CASES):
        good = [r for r in results if r['case'] == case['name']
                and r['label'] == 'good']
        bad  = [r for r in results if r['case'] == case['name']
                and r['label'] == 'bad']
        if not good or not bad:
            continue
        g, b = good[0], bad[0]
        line = (f'  {case["name"]:<32s} '
                f'{g["score_full"]:>8.4f} {b["score_full"]:>8.4f} '
                f'{g["score_full"]-b["score_full"]:>+8.4f}')
        if scorer_lite:
            line += (f' {g["score_lite"]:>8.4f} {b["score_lite"]:>8.4f} '
                     f'{g["score_lite"]-b["score_lite"]:>+8.4f} '
                     f'{g["time_full_ms"]:>7.1f}ms')
        else:
            line += f' {g["time_full_ms"]:>7.1f}ms'
        print(line)

    # Save JSON
    json_path = os.path.join(save_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {json_path}')

    return results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models', 'weights', 'simopa.pth')
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    s = SimOPAScorer(WEIGHT_PATH, backbone='resnet18', base_width=64,
                     device=DEVICE)
    pipe = PlacementScorer(s)
    run_tests(pipe, save_dir='./outputs')
