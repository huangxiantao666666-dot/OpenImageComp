"""
Object Placement Assistant

Two methods:
  - TopNet heatmap (fast):  one forward pass → 256x256 score map → top-K boxes
  - Grid + SimOPA (accurate): enumerate grid positions → composite → score each

Both can optionally apply Reinhard colour harmonization to results.

Usage:
    python app.py     # → http://127.0.0.1:7860
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
import gradio as gr

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__)) # get current file's dir
sys.path.insert(0, PROJ_ROOT) # when importing, look here first

from models import SimOPAScorer
from models.topnet import TopNetScorer
from pipeline import generate_candidates, PlacementScorer
from pipeline.harmonize import harmonize
from pipeline.auto_mask import auto_mask
from interpretability import generate_all

# ---------------------------------------------------------------------------
#  Models (lazy-load)
# ---------------------------------------------------------------------------
_simopa_scorer: PlacementScorer = None
_topnet_scorer: TopNetScorer = None
TOP_MODEL_PATH = os.path.join(PROJ_ROOT, '..', 'topnet_fixed',
    'checkpoints', 'buggy_best_weight.pth')
SIMOPA_PATH = os.path.join(PROJ_ROOT, 'models', 'weights', 'simopa.pth')


def get_simopa() -> PlacementScorer:
    global _simopa_scorer
    if _simopa_scorer is None:
        s = SimOPAScorer(SIMOPA_PATH, backbone='resnet18', base_width=64,
                         device='cuda')
        _simopa_scorer = PlacementScorer(s)
    return _simopa_scorer


def get_topnet() -> TopNetScorer:
    global _topnet_scorer
    if _topnet_scorer is None:
        if os.path.exists(TOP_MODEL_PATH):
            _topnet_scorer = TopNetScorer(TOP_MODEL_PATH, device='cuda')
        else:
            _topnet_scorer = TopNetScorer(
                os.path.join(PROJ_ROOT, 'models', 'weights', 'buggy_best_weight.pth'),
                device='cuda')
    return _topnet_scorer


# ---------------------------------------------------------------------------
#  Asset presets
# ---------------------------------------------------------------------------
ASSETS_DIR = os.path.join(PROJ_ROOT, 'assets')


def _list_assets(subdir: str, exclude_pattern: str = None) -> list:
    d = os.path.join(ASSETS_DIR, subdir)
    if not os.path.isdir(d):
        return []
    files = sorted(f for f in os.listdir(d)
                   if (f.endswith('.jpg') or f.endswith('.png'))
                   and (exclude_pattern is None or exclude_pattern not in f))
    return [(f, os.path.join(d, f)) for f in files]


def _find_mask_for(fg_name: str) -> str | None:
    base, ext = os.path.splitext(fg_name)
    p = os.path.join(ASSETS_DIR, 'foregrounds', f'{base}_mask{ext}')
    return p if os.path.exists(p) else None


def load_preset_bg(bg_file: str) -> np.ndarray | None:
    p = os.path.join(ASSETS_DIR, 'backgrounds', bg_file)
    if os.path.exists(p):
        return np.array(Image.open(p).convert('RGB'))
    return None


def load_preset_fg(fg_file: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    p = os.path.join(ASSETS_DIR, 'foregrounds', fg_file)
    if not os.path.exists(p):
        return None, None
    img = Image.open(p)
    if img.mode == 'RGBA':
        a = np.array(img); return a[:,:,:3], a[:,:,3]
    if img.mode == 'LA':
        a = np.array(img.convert('RGBA')); return a[:,:,:3], a[:,:,3]
    mp = _find_mask_for(fg_file)
    if mp:
        return np.array(img.convert('RGB')), np.array(Image.open(mp).convert('L'))
    return np.array(img.convert('RGB')), np.full((img.height, img.width), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
#  Preview
# ---------------------------------------------------------------------------
def _make_preview(bg_np: np.ndarray | None,
                  fg_np: np.ndarray | None) -> np.ndarray | None:
    """Stitch background and foreground side-by-side for preview."""
    if bg_np is None or fg_np is None:
        return None
    bg = Image.fromarray(bg_np).convert('RGB')
    fg = Image.fromarray(fg_np).convert('RGBA') if fg_np.shape[-1] == 4 \
         else Image.fromarray(fg_np).convert('RGB')

    # Match heights
    tg_h = max(bg.height, fg.height)
    ratio_bg = tg_h / bg.height
    ratio_fg = tg_h / fg.height
    bg_r = bg.resize((int(bg.width * ratio_bg), tg_h), Image.LANCZOS)
    fg_r = fg.resize((int(fg.width * ratio_fg), tg_h), Image.LANCZOS)

    # Side-by-side
    combined = Image.new('RGB', (bg_r.width + fg_r.width + 4, tg_h), (200, 200, 200))
    combined.paste(bg_r, (0, 0))
    combined.paste(fg_r, (bg_r.width + 4, 0))
    return np.array(combined)


# ---------------------------------------------------------------------------
#  Heatmap rendering
# ---------------------------------------------------------------------------
def _norm_heat(h):
    """Percentile-normalize heatmap [H,W] for better contrast."""
    lo, hi = np.percentile(h, 5), np.percentile(h, 95)
    if hi - lo > 0.001:
        return np.clip((h - lo) / (hi - lo), 0, 1)
    return h


def _topnet_heatmap_overlay(bg: Image.Image, hmap: np.ndarray) -> Image.Image:
    """Overlay a 256x256 TopNet heatmap onto the background."""
    bg_w, bg_h = bg.size
    h_full = np.array(Image.fromarray(
        (hmap * 255).astype(np.uint8)
    ).resize((bg_w, bg_h), Image.BILINEAR)).astype(np.float32) / 255.0
    h_full = _norm_heat(h_full)

    overlay = np.zeros((bg_h, bg_w, 4), dtype=np.uint8)
    overlay[:, :, 0] = np.clip(255 * (1.0 - h_full), 0, 255).astype(np.uint8)
    overlay[:, :, 1] = np.clip(255 * h_full, 0, 255).astype(np.uint8)
    overlay[:, :, 3] = (np.clip(h_full * 0.55 + 0.1, 0, 1) * 255).astype(np.uint8)

    bg_rgba = bg.convert('RGBA')
    return Image.alpha_composite(bg_rgba, Image.fromarray(overlay, 'RGBA')).convert('RGB')


def _grid_heatmap_overlay(bg: Image.Image, candidates: list,
                           sigma: float = 12.0) -> Image.Image:
    """Gaussian-interpolated score grid overlaid on background."""
    bg_w, bg_h = bg.size
    scale = 0.25
    h_s, w_s = int(bg_h * scale), int(bg_w * scale)
    smap = np.full((h_s, w_s), np.nan, dtype=np.float32)

    for c in candidates:
        cx = (c['bbox'][0] + c['bbox'][2]) // 2
        cy = (c['bbox'][1] + c['bbox'][3]) // 2
        sx, sy = int(cx * scale), int(cy * scale)
        if 0 <= sx < w_s and 0 <= sy < h_s:
            smap[sy, sx] = c['score']

    smap[np.isnan(smap)] = np.nanmean(smap)
    smap = gaussian_filter(smap, sigma=max(sigma * scale, 1.0))

    s_full = np.array(Image.fromarray(
        (smap * 255).astype(np.uint8)
    ).resize((bg_w, bg_h), Image.BILINEAR)).astype(np.float32) / 255.0
    s_full = _norm_heat(s_full)

    overlay = np.zeros((bg_h, bg_w, 4), dtype=np.uint8)
    overlay[:, :, 0] = np.clip(255 * (1.0 - s_full), 0, 255).astype(np.uint8)
    overlay[:, :, 1] = np.clip(255 * s_full, 0, 255).astype(np.uint8)
    overlay[:, :, 3] = (np.clip(s_full * 0.6 + 0.1, 0, 1) * 255).astype(np.uint8)

    return Image.alpha_composite(
        bg.convert('RGBA'),
        Image.fromarray(overlay, 'RGBA')
    ).convert('RGB')


# ---------------------------------------------------------------------------
#  Manual placement
# ---------------------------------------------------------------------------
def _manual_composite(bg: Image.Image, fg: Image.Image, fg_mask: Image.Image,
                      click_x: int, click_y: int, scale_pct: float,
                      rotate: float, use_harmony: bool,
                      use_shadow: bool = True) -> Image.Image:
    """Place fg CENTERED at (click_x, click_y) with scale and rotation."""
    # Scale foreground relative to background
    fg_w = int(bg.width * scale_pct / 100)
    fg_h = int(bg.height * scale_pct / 100)
    # Keep aspect ratio
    ratio = min(fg_w / max(fg.width, 1), fg_h / max(fg.height, 1), 2.0)
    fg_w = max(8, int(fg.width * ratio))
    fg_h = max(8, int(fg.height * ratio))

    # Resize
    fg_r = fg.resize((fg_w, fg_h), Image.LANCZOS)
    mk_r = fg_mask.resize((fg_w, fg_h), Image.LANCZOS)

    # Rotate
    if abs(rotate) > 0.1:
        fg_r = fg_r.rotate(rotate, resample=Image.BICUBIC, expand=True)
        mk_r = mk_r.rotate(rotate, resample=Image.BICUBIC, expand=True)

    # Position: centre of fg at click point
    x1 = click_x - fg_r.width // 2
    y1 = click_y - fg_r.height // 2
    # Clip to bg boundaries
    x1 = max(0, min(x1, bg.width - 1))
    y1 = max(0, min(y1, bg.height - 1))
    x2 = min(x1 + fg_r.width, bg.width)
    y2 = min(y1 + fg_r.height, bg.height)
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return bg.copy()

    bg_arr = np.array(bg.copy()).astype(np.float32)
    fg_arr = np.array(fg_r.crop((0, 0, pw, ph))).astype(np.float32)
    mk_arr = np.array(mk_r.crop((0, 0, pw, ph))).astype(np.float32) / 255.0
    if mk_arr.ndim == 2:
        mk_arr = np.stack([mk_arr] * 3, axis=-1)

    bg_arr[y1:y2, x1:x2] = fg_arr * mk_arr + bg_arr[y1:y2, x1:x2] * (1 - mk_arr)
    comp = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8))

    if use_harmony:
        full_mask = np.zeros((bg.height, bg.width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = (np.array(mk_r.crop((0, 0, pw, ph))) > 127).astype(np.uint8) * 255
        comp = harmonize(comp, Image.fromarray(full_mask, 'L'))

    if use_shadow:
        from pipeline.shadow import add_shadow
        bbox_mask = np.zeros((bg.height, bg.width), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 255
        comp = add_shadow(comp, Image.fromarray(bbox_mask),  # no mode= kw
                          [x1, y1, x2, y2])

    return comp


def manual_place(bg_np, fg_rgb_np, fg_mask_np,
                 click_x, click_y, scale_pct, rotate_slider,
                 use_harmony, progress=gr.Progress()):
    """Click-to-place: composite fg centered at click point → score."""
    if bg_np is None or fg_rgb_np is None:
        return None, 'Upload both images, then click on the background to place.'

    bg = Image.fromarray(bg_np).convert('RGB')
    if fg_rgb_np.shape[-1] == 4:
        fg = Image.fromarray(fg_rgb_np[:, :, :3]).convert('RGB')
        fg_mask = Image.fromarray(fg_rgb_np[:, :, 3]).convert('L')
    else:
        fg = Image.fromarray(fg_rgb_np).convert('RGB')
        if fg_mask_np is not None:
            fg_mask = Image.fromarray(fg_mask_np).convert('L')
        else:
            mask_arr = auto_mask(fg_rgb_np)
            fg_mask = Image.fromarray(mask_arr).convert('L')

    comp = _manual_composite(bg, fg, fg_mask,
                              int(click_x), int(click_y),
                              scale_pct, rotate_slider, use_harmony)

    # Score with SimOPA
    from pipeline.composite import make_composite
    fg_w = int(bg.width * scale_pct / 100)
    fg_h = int(bg.height * scale_pct / 100)
    ratio = min(fg_w / max(fg.width, 1), fg_h / max(fg.height, 1), 2.0)
    fg_w = max(8, int(fg.width * ratio))
    fg_h = max(8, int(fg.height * ratio))
    x1 = int(click_x) - fg_w // 2
    y1 = int(click_y) - fg_h // 2
    bbox = [x1, y1, x1 + fg_w, y1 + fg_h]
    _, full_mask = make_composite(bg, fg, fg_mask, bbox)
    score = get_simopa().model.score(comp, full_mask)

    status = (f'**SimOPA**: {score:.4f}  |  '
              f'click=({int(click_x)},{int(click_y)})  '
              f'scale={scale_pct:.0f}%  rot={rotate_slider:.0f}deg'
              + ('  +harm' if use_harmony else ''))

    return comp, status


def compute_mask(fg_rgb_np, mask_source, mask_upload):
    """Run mask generation based on user-selected source."""
    if mask_source == 'alpha' and fg_rgb_np is not None and fg_rgb_np.shape[-1] == 4:
        return fg_rgb_np[:, :, 3]
    if mask_source == 'upload' and mask_upload is not None:
        return mask_upload
    if mask_source == 'sam2':
        return auto_mask(fg_rgb_np, prefer_sam=True)
    if mask_source == 'opencv':
        return auto_mask(fg_rgb_np, prefer_sam=False)
    # fallback
    return auto_mask(fg_rgb_np, prefer_sam=False)


# ---------------------------------------------------------------------------
#  Analysis  (auto-search: TopNet / Grid)
# ---------------------------------------------------------------------------
def analyze(bg_np: np.ndarray | None,
            fg_rgb_np: np.ndarray | None,
            fg_mask_np: np.ndarray | None,
            method: str,
            grid_size: float, n_scales: float,
            topnet_scale: float,
            use_harmony: bool,
            mask_source: str = 'sam2',
            score_mode: str = '4ch',
            progress=gr.Progress()) -> tuple:
    """
    Args:
        method: 'topnet' | 'grid'

    Returns: (heatmap_img, gallery, table, status_md)
    """
    if bg_np is None or fg_rgb_np is None:
        return None, [], [], {}, 'Please upload/select both images.'

    def _p(val, desc=''):
        if progress is not None:
            progress(val, desc=desc)

    _p(0.05, desc='Preparing images ...')
    bg = Image.fromarray(bg_np).convert('RGB')
    if fg_rgb_np.shape[-1] == 4:
        fg = Image.fromarray(fg_rgb_np[:,:,:3]).convert('RGB')
        fg_mask = Image.fromarray(fg_rgb_np[:,:,3]).convert('L')
    else:
        fg = Image.fromarray(fg_rgb_np).convert('RGB')
        if fg_mask_np is not None:
            fg_mask = Image.fromarray(fg_mask_np).convert('L')
        else:
            # Auto-mask via selected source (SAM2 or OpenCV)
            mask_arr = compute_mask(fg_rgb_np, mask_source, None)
            fg_mask = Image.fromarray(mask_arr).convert('L')

    candidates = []
    topnet_hmap = None
    simopa = get_simopa()

    if method == 'topnet':
        # ---- TopNet: single forward pass → heatmap → top-K boxes ----
        _p(0.10, desc='Running TopNet (single forward pass) ...')
        t0 = time.time()

        # Scale foreground to user-selected proportion of background
        from pipeline.auto_mask import auto_mask as _am_fn
        fg_scaled = fg.copy()
        fg_mask_scaled = fg_mask.copy()
        target_w = int(bg.width * topnet_scale / 100)
        target_h = int(bg.height * topnet_scale / 100)
        # Keep aspect ratio
        ratio = min(target_w / max(fg.width, 1), target_h / max(fg.height, 1), 1.5)
        new_w = max(8, int(fg.width * ratio))
        new_h = max(8, int(fg.height * ratio))
        fg_scaled = fg.resize((new_w, new_h), Image.LANCZOS)
        fg_mask_scaled = fg_mask.resize((new_w, new_h), Image.LANCZOS)

        topnet = get_topnet()
        hmap = topnet.heatmap(bg, fg_scaled, fg_mask_scaled)
        # Use the scaled fg for compositing
        fg_use, fg_mask_use = fg_scaled, fg_mask_scaled
        t_topnet = time.time() - t0

        _p(0.25, desc='Extracting local maxima ...')
        boxes = topnet.top_k_boxes(bg, fg_use, fg_mask_use, k=15,
                                    fg_w=fg_use.width, fg_h=fg_use.height)
        topnet_hmap = hmap

        _p(0.30, desc=f'SimOPA fine-scoring {len(boxes)} candidates ...')
        for b in boxes:
            from pipeline.composite import make_composite
            comp, cmask = make_composite(bg, fg_use, fg_mask_use, b['bbox'])
            s = simopa.model.score(comp, cmask, mode=score_mode, bbox=b['bbox'])
            candidates.append({
                'bbox':      b['bbox'],
                'score':     s,
                'topnet':    b['score'],
                'composite': comp,
                'mask':      cmask,
                'scale':     1.0,
            })
        candidates.sort(key=lambda r: r['score'], reverse=True)
        t_total = time.time() - t0

    else:
        # ---- Grid: enumerate → composite → score each with SimOPA ----
        _p(0.10, desc='Generating candidate grid ...')
        t0 = time.time()
        grid = int(grid_size)
        sc = int(n_scales)
        raw = generate_candidates(bg.width, bg.height,
                                   fg.width, fg.height,
                                   grid_size=grid, n_scales=sc)
        if not raw:
            for fb in [5, 7, 9]:
                raw = generate_candidates(bg.width, bg.height,
                                           fg.width, fg.height,
                                           grid_size=grid, n_scales=fb)
                if raw: break
        if not raw:
            return None, [], [], {}, 'Foreground too large.'

        _p(0.15, desc=f'Scoring {len(raw)} candidates with SimOPA ...')
        candidates = simopa.score_candidates(bg, fg, fg_mask, raw,
                                               score_mode=score_mode)
        t_total = time.time() - t0

    # --- Common: build heatmap, gallery, table ---
    _p(0.70, desc='Building heatmap ...')
    if topnet_hmap is not None:
        heatmap = _topnet_heatmap_overlay(bg, topnet_hmap)
    else:
        heatmap = _grid_heatmap_overlay(bg, candidates)

    _p(0.85, desc='Assembling gallery ...')
    top_k = min(5, len(candidates))
    gallery = []
    for i, r in enumerate(candidates[:top_k]):
        comp = r['composite']
        if use_harmony:
            comp = harmonize(comp, r['mask'])
        extra = ''
        if 'topnet' in r:
            extra = f'  TN={r["topnet"]:.3f}'
        label = (f'#{i+1}  SimOPA={r["score"]:.3f}{extra}'
                 f'  s={r.get("scale",1.0):.1f}'
                 + (' (harm)' if use_harmony else ''))
        gallery.append((comp, label))

    _p(0.95, desc='Building table ...')
    cols = ['Rank', 'SimOPA', 'Scale', 'Position (x1,y1,x2,y2)']
    if method == 'topnet':
        cols.insert(2, 'TopNet')
    table = [cols]
    bg_w, bg_h = bg.size
    for i, r in enumerate(candidates[:15]):
        row = [str(i+1), f'{r["score"]:.4f}']
        if method == 'topnet':
            row.append(f'{r.get("topnet","?"):.4f}')
        row.append(f'{r.get("scale",1.0):.2f}')
        b = r['bbox']
        row.append(f'({b[0]/bg_w:.2f},{b[1]/bg_h:.2f},{b[2]/bg_w:.2f},{b[3]/bg_h:.2f})')
        table.append(row)

    best = candidates[0]
    n = len(candidates)
    method_label = 'TopNet + SimOPA' if method == 'topnet' else 'Grid + SimOPA'
    # Interpretability on top-1 composite
    interp = {}
    if candidates and 'composite' in candidates[0] and 'mask' in candidates[0]:
        try:
            interp = generate_all(simopa.model.model, candidates[0]['composite'],
                                  candidates[0]['mask'])
        except Exception as e:
            print(f'[Interpretation] Failed: {e}')

    status = (
        f'**Method**: {method_label}  |  '
        f'**{n} candidates** in {t_total:.2f}s  |  '
        f'**Best**: SimOPA={best["score"]:.3f}'
        + (f'  TopNet={best.get("topnet","?"):.3f}' if 'topnet' in best else '')
        + f'  pos=({best["bbox"][0]/bg_w:.2f},{best["bbox"][1]/bg_h:.2f})'
    )
    return heatmap, gallery, table, interp, status


# ---------------------------------------------------------------------------
#  UI
# ---------------------------------------------------------------------------
def build_ui():
    bg_choices = [f[0] for f in _list_assets('backgrounds')]
    fg_choices = [f[0] for f in _list_assets('foregrounds', exclude_pattern='_mask')]
    has_presets = bool(bg_choices) and bool(fg_choices)

    with gr.Blocks(title='Object Placement Assistant') as demo:
        gr.Markdown("""
        # Object Placement Assistant

        **Two methods available**:
        - **TopNet** (CVPR 2023): one forward pass → 256×256 heatmap → top-K boxes + SimOPA fine-scoring (fast)
        - **Grid + SimOPA**: enumerate grid positions, composite + score each (accurate)
        - Optional: **Reinhard colour harmonization** on results
        """)

        with gr.Tabs():
            if has_presets:
                with gr.Tab('Presets'):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=290):
                            gr.Markdown('### Preset images')
                            p_bg = gr.Dropdown(choices=bg_choices, value=bg_choices[0],
                                                label='Background')
                            p_fg = gr.Dropdown(choices=fg_choices, value=fg_choices[0],
                                                label='Foreground (mask auto-loaded)')
                            preview_p = gr.Image(label='Preview (bg + fg)',
                                                  type='numpy', height=220)

                            with gr.Accordion('Options', open=True):
                                p_method = gr.Radio(
                                    choices=['topnet', 'grid'],
                                    value='topnet', label='Placement method',
                                    info='TopNet = single-pass heatmap; Grid = enumerate + SimOPA score')
                                p_mask_src = gr.Radio(
                                    choices=['sam2', 'opencv', 'alpha'],
                                    value='alpha', label='Mask source',
                                    info='SAM2 / OpenCV / alpha channel (presets have alpha)')
                                p_tn_scale = gr.Slider(5, 100, value=30, step=5,
                                                        label='FG scale for TopNet (%)',
                                                        info='Foreground size as % of background')
                                p_grid = gr.Slider(3, 9, value=5, step=1,
                                                    label='Grid density (Grid method only)')
                                p_scales = gr.Slider(1, 7, value=5, step=1,
                                                      label='Number of scales (Grid method only)')
                                p_score_m = gr.Radio(choices=['4ch', '3ch', 'crop'], value='4ch',
                                                       label='SimOPA scoring mode',
                                                       info='4ch=RGB+mask, 3ch=RGB only, crop=local area')
                                p_harmony = gr.Checkbox(
                                    value=False, label='Apply colour harmonization',
                                    info='Reinhard/PCTNet colour transfer')

                            btn_p = gr.Button('Analyze', variant='primary', size='lg')
                            status_p = gr.Markdown('')
                            bg_st = gr.State(); fg_st = gr.State(); mk_st = gr.State()

                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.Tab('Heatmap'):
                                    gr.Markdown('*Green = high score, Red = low*')
                                    hm_p = gr.Image(label='Placement Heatmap', type='pil', height=380)
                                with gr.Tab('Top Composites'):
                                    gal_p = gr.Gallery(label='Top-5 Placements',
                                                        columns=5, rows=2,
                                                        object_fit='contain', height=500)
                                with gr.Tab('Detail Table'):
                                    tbl_p = gr.Dataframe(label='Scores')
                                with gr.Tab('Interpretation'):
                                    with gr.Tabs():
                                        with gr.Tab('Grad-CAM'):
                                            i_gc = gr.Image(label='Grad-CAM', type='pil', height=300)
                                        with gr.Tab('Saliency'):
                                            i_sal = gr.Image(label='Saliency Map', type='pil', height=300)
                                        with gr.Tab('Occlusion'):
                                            i_occ = gr.Image(label='Occlusion Map', type='pil', height=300)
                                        with gr.Tab('Features L2'):
                                            i_f2 = gr.Image(label='Layer2 Features', type='pil', height=300)
                                        with gr.Tab('Features L4'):
                                            i_f4 = gr.Image(label='Layer4 Features', type='pil', height=300)

                    def _on_change(bg_f, fg_f):
                        bg = load_preset_bg(bg_f)
                        fg, mk = load_preset_fg(fg_f)
                        preview = _make_preview(bg, fg)
                        return bg, fg, mk, preview

                    for w in [p_bg, p_fg]:
                        w.change(_on_change, [p_bg, p_fg],
                                 [bg_st, fg_st, mk_st, preview_p])

                    def _unpack_interp(hm, gal, tbl, interp, st):
                        return (hm, gal, tbl,
                                interp.get('gradcam'), interp.get('saliency'),
                                interp.get('occlusion'),
                                interp.get('features_layer2'),
                                interp.get('features_layer4'), st)

                    btn_p.click(
                        lambda *args: _unpack_interp(*analyze(*args)),
                        [bg_st, fg_st, mk_st, p_method,
                         p_grid, p_scales, p_tn_scale, p_harmony, p_mask_src, p_score_m],
                        [hm_p, gal_p, tbl_p, i_gc, i_sal, i_occ, i_f2, i_f4, status_p])

            # ---- Custom Upload ----
            with gr.Tab('Custom Upload'):
                with gr.Row():
                    with gr.Column(scale=1, min_width=290):
                        gr.Markdown('### Upload your images')
                        u_bg = gr.Image(label='Background', type='numpy', height=180)
                        u_fg = gr.Image(label='Foreground (transparent PNG = auto-mask)',
                                         type='numpy', height=180)
                        u_mask = gr.Image(label='Mask (optional — auto-generated if empty)',
                                           type='numpy', height=120)

                        with gr.Accordion('Options', open=True):
                            u_method = gr.Radio(
                                choices=['topnet', 'grid'],
                                value='topnet', label='Placement method')
                            u_mask_src = gr.Radio(
                                choices=['sam2', 'opencv', 'alpha'],
                                value='sam2', label='Mask source',
                                info='SAM2 / OpenCV / alpha channel')
                            u_tn_scale = gr.Slider(5, 100, value=30, step=5,
                                                    label='FG scale for TopNet (%)')
                            u_grid = gr.Slider(3, 9, value=5, step=1,
                                                label='Grid density (Grid method only)')
                            u_scales = gr.Slider(1, 7, value=5, step=1,
                                                  label='Number of scales (Grid method only)')
                            u_score_m = gr.Radio(choices=['4ch', '3ch', 'crop'], value='4ch',
                                                   label='SimOPA scoring mode',
                                                   info='4ch=RGB+mask, 3ch=RGB only, crop=local area')
                            u_harmony = gr.Checkbox(
                                value=False, label='Apply colour harmonization')

                        btn_u = gr.Button('Analyze', variant='primary', size='lg')
                        status_u = gr.Markdown('')

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab('Heatmap'):
                                gr.Markdown('*Green = high score, Red = low*')
                                hm_u = gr.Image(label='Placement Heatmap', type='pil', height=380)
                            with gr.Tab('Top Composites'):
                                gal_u = gr.Gallery(label='Top-5 Placements',
                                                    columns=5, rows=2,
                                                    object_fit='contain', height=420)
                            with gr.Tab('Detail Table'):
                                tbl_u = gr.Dataframe(label='Scores')

                btn_u.click(analyze,
                            [u_bg, u_fg, u_mask, u_method,
                             u_grid, u_scales, u_tn_scale, u_harmony, u_mask_src, u_score_m],
                            [hm_u, gal_u, tbl_u, status_u])

            # ============================================================
            #  Tab 3: Manual Placement
            # ============================================================
            with gr.Tab('Manual Placement'):
                gr.Markdown('### Click-to-Place — click on the background to place the foreground')
                with gr.Row():
                    with gr.Column(scale=1, min_width=290):
                        m_src = gr.Radio(choices=['preset', 'upload'], value='preset',
                                          label='Image source')
                        with gr.Group(visible=True) as m_preset_grp:
                            m_bg_sel = gr.Dropdown(
                                choices=bg_choices, value=bg_choices[0] if bg_choices else None,
                                label='Background')
                            m_fg_sel = gr.Dropdown(
                                choices=fg_choices, value=fg_choices[0] if fg_choices else None,
                                label='Foreground')
                        with gr.Group(visible=False) as m_upload_grp:
                            m_bg_up = gr.Image(label='Upload BG', type='numpy', height=150)
                            m_fg_up = gr.Image(label='Upload FG', type='numpy', height=150)

                        gr.Markdown('**Mask source** (if fg has no alpha channel)')
                        m_mask_src = gr.Radio(
                            choices=['sam2', 'opencv', 'alpha', 'upload'],
                            value='sam2', label='Segmentation method')
                        m_mask_up = gr.Image(label='Upload mask (if source=upload)',
                                              type='numpy', height=100)

                        with gr.Accordion('Placement controls', open=True):
                            m_scale = gr.Slider(5, 80, value=25, step=1,
                                                 label='FG scale (% of background)')
                            m_rot = gr.Slider(-180, 180, value=0, step=1,
                                               label='Rotation (degrees)')
                            with gr.Row():
                                m_shadow = gr.Checkbox(value=True, label='Shadow',
                                                        info='Soft drop shadow below object')
                                m_harmony = gr.Checkbox(value=False, label='Harmonize',
                                                         info='PCTNet/Reinhard')

                        with gr.Row():
                            btn_place = gr.Button('Place Object', variant='primary', size='lg')
                            btn_undo = gr.Button('Undo Last', variant='secondary')
                            btn_reset = gr.Button('Reset All', variant='secondary')
                        status_m = gr.Markdown('Select images then click on the background. '
                                                'Place multiple objects by clicking different locations.')

                        m_bg_st = gr.State(); m_orig_bg_st = gr.State()
                        m_fg_st = gr.State(); m_mk_st = gr.State()
                        m_click_x = gr.State(0); m_click_y = gr.State(0)
                        m_placed = gr.State([])

                    with gr.Column(scale=2):
                        comp_m = gr.Image(label='Click on the image to place foreground',
                                           type='pil', height=450,
                                           elem_id='manual-bg')

                        mask_preview = gr.Image(label='Foreground Mask Preview',
                                                 type='numpy', height=150)

                # --- Manual placement wiring ---
                m_src.change(lambda s: (gr.update(visible=(s == 'preset')),
                                         gr.update(visible=(s == 'upload'))),
                             [m_src], [m_preset_grp, m_upload_grp])

                # BG change → full reset
                def _load_bg(src, sel_bg, up_bg):
                    if src == 'preset':
                        bg = load_preset_bg(sel_bg)
                    else:
                        bg = up_bg
                    bg_pil = Image.fromarray(bg).convert('RGB') if bg is not None else None
                    return bg, bg, bg_pil, []  # bg_st, orig_bg_st, comp_m, placed

                for w in [m_src, m_bg_sel, m_bg_up]:
                    w.change(_load_bg, [m_src, m_bg_sel, m_bg_up],
                             [m_bg_st, m_orig_bg_st, comp_m, m_placed])

                # FG change → only update fg + mask, keep composite as-is
                def _load_fg(src, sel_fg, up_fg, mask_src, mask_up):
                    if src == 'preset':
                        fg_arr, mk = load_preset_fg(sel_fg)
                    elif up_fg is not None:
                        fg_arr = up_fg[:,:,:3] if up_fg.shape[-1]==4 else up_fg
                        mk = up_fg[:,:,3] if up_fg.shape[-1]==4 else None
                    else:
                        return None, None, None
                    if fg_arr is not None and (mask_src in ('sam2', 'opencv') or mk is None):
                        mk = compute_mask(fg_arr, mask_src, mask_up)
                    return fg_arr, mk, mk

                for w in [m_fg_sel, m_fg_up, m_mask_src, m_mask_up]:
                    w.change(_load_fg,
                             [m_src, m_fg_sel, m_fg_up, m_mask_src, m_mask_up],
                             [m_fg_st, m_mk_st, mask_preview])

                # Click → place fg at click position, composite onto current bg
                def _place_at(bg_np, fg_np, mk_np, cx, cy, scale, rot, harm, shadow, placed):
                    if bg_np is None or fg_np is None:
                        return None, placed, None, 'Load images first.'
                    if cx <= 0 and cy <= 0:
                        return None, placed, None, 'Click on the background image first.'
                    bg = Image.fromarray(bg_np).convert('RGB')
                    if fg_np.shape[-1] == 4:
                        fg = Image.fromarray(fg_np[:,:,:3]).convert('RGB')
                        fg_mask = Image.fromarray(fg_np[:,:,3]).convert('L')
                    else:
                        fg = Image.fromarray(fg_np).convert('RGB')
                        fg_mask = (Image.fromarray(mk_np).convert('L')
                                   if mk_np is not None
                                   else Image.new('L', fg.size, 255))
                    comp = _manual_composite(bg, fg, fg_mask, cx, cy,
                                              scale, rot, harm, shadow)
                    # Score with SimOPA
                    from pipeline.composite import make_composite
                    bb = [cx - fg.width//2, cy - fg.height//2,
                          cx + fg.width//2, cy + fg.height//2]
                    _, fm = make_composite(bg, fg, fg_mask, bb)
                    score = get_simopa().model.score(comp, fm)
                    new_obj = {'cx': cx, 'cy': cy, 'scale': scale, 'rot': rot,
                               'harm': harm, 'shadow': shadow, 'score': score,
                               'fg': np.array(fg), 'mk': np.array(fg_mask)}
                    comp_arr = np.array(comp)
                    n_objs = len(placed)
                    return comp, placed, comp_arr, \
                        f'#{n_objs} SimOPA={score:.3f} at ({cx},{cy}) [{n_objs} obj(s)]'

                def _on_click(bg, fg, mk, s, r, h, sh, pl, evt: gr.SelectData):
                    comp_, placed_, bg_arr_, status_ = _place_at(
                        bg, fg, mk, evt.index[0], evt.index[1], s, r, h, sh, pl)
                    return comp_, placed_, bg_arr_, status_, evt.index[0], evt.index[1]

                comp_m.select(
                    fn=_on_click,
                    inputs=[m_bg_st, m_fg_st, m_mk_st,
                            m_scale, m_rot, m_harmony, m_shadow, m_placed],
                    outputs=[comp_m, m_placed, m_bg_st, status_m, m_click_x, m_click_y])

                btn_place.click(
                    fn=_place_at,
                    inputs=[m_bg_st, m_fg_st, m_mk_st,
                            m_click_x, m_click_y,
                            m_scale, m_rot, m_harmony, m_shadow, m_placed],
                    outputs=[comp_m, m_placed, m_bg_st, status_m])

                # Reset — restore original bg, clear placed objects
                def _reset(bg_np):
                    if bg_np is None: return None, [], bg_np
                    from PIL import Image as _Img
                    return _Img.fromarray(bg_np).convert('RGB'), [], bg_np
                btn_reset.click(_reset, [m_orig_bg_st], [comp_m, m_placed, m_bg_st])

                # Undo — remove last object, go back to original bg
                def _undo(bg_np, placed):
                    if not placed:
                        return None, placed, bg_np, 'Nothing to undo.'
                    placed.pop()
                    bg = Image.fromarray(bg_np).convert('RGB') if bg_np is not None else None
                    if bg is None:
                        return None, placed, bg_np, 'No background.'
                    n = len(placed)
                    return bg, placed, bg_np, f'Removed last object. {n} remaining.' \
                        if n > 0 else 'All objects removed.'
                btn_undo.click(_undo, [m_orig_bg_st, m_placed],
                               [comp_m, m_placed, m_bg_st, status_m])

        gr.Markdown("""
        ---
        **Models**: TopNet (113M, CVPR 2023) + SimOPA (11M) |
        **Harmonization**: Reinhard et al. (2001) colour transfer
        """)

    return demo


if __name__ == '__main__':
    get_simopa()
    # Try to load TopNet if available
    if os.path.exists(TOP_MODEL_PATH):
        get_topnet()
    demo = build_ui()
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
