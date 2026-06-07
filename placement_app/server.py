"""
REST API server for ImageCompose Android app.

Usage:
    python server.py --port 8000
    # Health check:  http://<host>:8000/api/health
    # Docs:          http://<host>:8000/docs
"""

import io, os, sys, time, base64, argparse
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from models import SimOPAScorer
from models.topnet import TopNetScorer
from pipeline import generate_candidates, PlacementScorer
from pipeline.composite import make_composite
from pipeline.harmonize import harmonize
from pipeline.auto_mask import auto_mask

# ---------------------------------------------------------------------------
#  Model paths
# ---------------------------------------------------------------------------
SIMOPA_PATH = os.path.join(PROJ_ROOT, 'models', 'weights', 'simopa.pth')
TOP_MODEL_PATH = os.path.join(
    PROJ_ROOT, '..', 'topnet_fixed', 'checkpoints', 'buggy_best_weight.pth')
PCTNET_PATH = os.path.join(PROJ_ROOT, 'models', 'weights', 'PCTNet.pth')

# ---------------------------------------------------------------------------
#  Lazy model loading
# ---------------------------------------------------------------------------
_simopa_scorer = None
_topnet_scorer = None


def get_simopa():
    global _simopa_scorer
    if _simopa_scorer is None:
        s = SimOPAScorer(SIMOPA_PATH, base_width=64, device='cuda')
        _simopa_scorer = PlacementScorer(s)
    return _simopa_scorer


def get_topnet():
    global _topnet_scorer
    if _topnet_scorer is None:
        if os.path.exists(TOP_MODEL_PATH):
            _topnet_scorer = TopNetScorer(TOP_MODEL_PATH, device='cuda')
        else:
            alt = os.path.join(PROJ_ROOT, 'models', 'weights', 'best_weight.pth')
            if os.path.exists(alt):
                _topnet_scorer = TopNetScorer(alt, device='cuda')
    return _topnet_scorer


def _img_to_base64(img: Image.Image, fmt: str = 'JPEG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _read_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)))


# ---------------------------------------------------------------------------
#  App
# ---------------------------------------------------------------------------
app = FastAPI(title='ImageCompose API', version='1.0')


@app.get('/api/health')
async def health():
    models_loaded = ['simopa']
    models_missing = []
    if get_topnet() is not None:
        models_loaded.append('topnet')
    else:
        models_missing.append('topnet')
    if os.path.exists(PCTNET_PATH):
        models_loaded.append('pctnet')
    else:
        models_missing.append('pctnet')
    return {
        'status': 'ok',
        'models_loaded': models_loaded,
        'models_missing': models_missing,
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
        'version': '1.0',
    }


# ===================================================================
#  POST /api/place  — auto-search (TopNet / Grid)
# ===================================================================
@app.post('/api/place')
async def api_place(
    bg: UploadFile = File(...),
    fg: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    method: str = Form('topnet'),
    fg_scale: float = Form(30.0),
    top_k: int = Form(5),
    harmonize_flag: bool = Form(False),
    max_dim: int = Form(512),
):
    bg_np = _read_image(await bg.read())
    fg_np = _read_image(await fg.read())
    bg_pil = Image.fromarray(bg_np).convert('RGB')

    # Foreground + mask
    if fg_np.shape[-1] == 4:
        fg_rgb = fg_np[:, :, :3]
        fg_mask = fg_np[:, :, 3]
    else:
        fg_rgb = fg_np
        if mask is not None:
            mk_np = _read_image(await mask.read())
            fg_mask = mk_np if mk_np.ndim == 2 else mk_np[:, :, 0]
        else:
            fg_mask = auto_mask(fg_rgb, prefer_sam=True)

    fg_pil = Image.fromarray(fg_rgb).convert('RGB')
    fg_mk_pil = Image.fromarray(fg_mask).convert('L')

    # Resize if too large
    if max(bg_pil.size) > max_dim:
        ratio = max_dim / max(bg_pil.size)
        new_sz = (int(bg_pil.width * ratio), int(bg_pil.height * ratio))
        bg_pil = bg_pil.resize(new_sz, Image.LANCZOS)

    t0 = time.time()
    simopa = get_simopa()
    candidates = []

    if method == 'topnet':
        topnet = get_topnet()
        if topnet is None:
            return JSONResponse({'status': 'error', 'message': 'TopNet model not loaded'}, 500)
        # Scale fg
        tw, th = int(bg_pil.width * fg_scale / 100), int(bg_pil.height * fg_scale / 100)
        r = min(tw / max(fg_pil.width, 1), th / max(fg_pil.height, 1), 1.5)
        fg_s = fg_pil.resize((max(8, int(fg_pil.width * r)), max(8, int(fg_pil.height * r))), Image.LANCZOS)
        mk_s = fg_mk_pil.resize(fg_s.size, Image.LANCZOS)

        hmap = topnet.heatmap(bg_pil, fg_s, mk_s)
        boxes = topnet.top_k_boxes(bg_pil, fg_s, mk_s, k=top_k * 3,
                                    fg_w=fg_s.width, fg_h=fg_s.height)
        for b in boxes:
            comp, cmask = make_composite(bg_pil, fg_s, mk_s, b['bbox'])
            score = simopa.model.score(comp, cmask)
            candidates.append({'bbox': b['bbox'], 'score': score, 'topnet': b['score'],
                               'composite': comp, 'mask': cmask, 'scale': 1.0})
    else:
        raw = generate_candidates(bg_pil.width, bg_pil.height,
                                   fg_pil.width, fg_pil.height,
                                   grid_size=5, n_scales=5)
        candidates = simopa.score_candidates(bg_pil, fg_pil, fg_mk_pil, raw)

    candidates.sort(key=lambda r: r['score'], reverse=True)
    candidates = candidates[:top_k]
    elapsed = time.time() - t0

    # ---- Build heatmap ----
    heatmap_img = _draw_heatmap_server(bg_pil, candidates)

    # ---- Build response ----
    composites = []
    for i, r in enumerate(candidates):
        comp = r['composite']
        if harmonize_flag:
            comp = harmonize(comp, r['mask'])
        composites.append({
            'rank': i + 1,
            'score_simopa': round(r['score'], 4),
            'score_topnet': round(r.get('topnet', 0), 4) if 'topnet' in r else None,
            'bbox': [r['bbox'][0]/bg_pil.width, r['bbox'][1]/bg_pil.height,
                      r['bbox'][2]/bg_pil.width, r['bbox'][3]/bg_pil.height],
            'scale': round(r.get('scale', 1.0), 2),
            'image': _img_to_base64(comp, 'JPEG'),
        })

    # Table
    cols = ['Rank', 'SimOPA', 'Scale', 'Position']
    if method == 'topnet':
        cols.insert(2, 'TopNet')
    rows = []
    for i, r in enumerate(candidates):
        row = [str(i + 1), f'{r["score"]:.4f}']
        if method == 'topnet':
            row.append(f'{r.get("topnet", 0):.4f}')
        row.append(f'{r.get("scale", 1.0):.2f}')
        b = r['bbox']
        row.append(f'({b[0]/bg_pil.width:.2f},{b[1]/bg_pil.height:.2f},'
                   f'{b[2]/bg_pil.width:.2f},{b[3]/bg_pil.height:.2f})')
        rows.append(row)

    return {
        'status': 'ok',
        'method': method,
        'total_candidates': len(candidates),
        'elapsed_seconds': round(elapsed, 2),
        'heatmap': _img_to_base64(heatmap_img, 'JPEG'),
        'composites': composites,
        'table': {'columns': cols, 'rows': rows},
    }


# ===================================================================
#  POST /api/place_manual  — manual click-to-place
# ===================================================================
@app.post('/api/place_manual')
async def api_place_manual(
    bg: UploadFile = File(...),
    fg: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    x: int = Form(...),
    y: int = Form(...),
    scale: float = Form(30.0),
    rotation: float = Form(0.0),
    harmonize_flag: bool = Form(False),
):
    bg_np = _read_image(await bg.read())
    fg_np = _read_image(await fg.read())
    bg_pil = Image.fromarray(bg_np).convert('RGB')

    if fg_np.shape[-1] == 4:
        fg_rgb = fg_np[:, :, :3]
        fg_mask = fg_np[:, :, 3]
    else:
        fg_rgb = fg_np
        if mask is not None:
            mk_np = _read_image(await mask.read())
            fg_mask = mk_np if mk_np.ndim == 2 else mk_np[:, :, 0]
        else:
            fg_mask = auto_mask(fg_rgb, prefer_sam=True)

    fg_pil = Image.fromarray(fg_rgb).convert('RGB')
    fg_mk_pil = Image.fromarray(fg_mask).convert('L')

    # Scale
    fg_w = int(bg_pil.width * scale / 100)
    fg_h = int(bg_pil.height * scale / 100)
    ratio = min(fg_w / max(fg_pil.width, 1), fg_h / max(fg_pil.height, 1), 2.0)
    fg_w = max(8, int(fg_pil.width * ratio))
    fg_h = max(8, int(fg_pil.height * ratio))
    fg_s = fg_pil.resize((fg_w, fg_h), Image.LANCZOS)
    mk_s = fg_mk_pil.resize((fg_w, fg_h), Image.LANCZOS)

    # Rotate
    if abs(rotation) > 0.1:
        fg_s = fg_s.rotate(rotation, resample=Image.BICUBIC, expand=True)
        mk_s = mk_s.rotate(rotation, resample=Image.BICUBIC, expand=True)

    # Centre at (x, y)
    x1 = x - fg_s.width // 2
    y1 = y - fg_s.height // 2
    x1 = max(0, min(x1, bg_pil.width - 1))
    y1 = max(0, min(y1, bg_pil.height - 1))
    x2 = min(x1 + fg_s.width, bg_pil.width)
    y2 = min(y1 + fg_s.height, bg_pil.height)
    pw, ph = x2 - x1, y2 - y1

    bg_arr = np.array(bg_pil).astype(np.float32)
    fg_arr = np.array(fg_s.crop((0, 0, pw, ph))).astype(np.float32)
    mk_arr = np.array(mk_s.crop((0, 0, pw, ph))).astype(np.float32) / 255.0
    if mk_arr.ndim == 2:
        mk_arr = np.stack([mk_arr] * 3, axis=-1)
    bg_arr[y1:y2, x1:x2] = fg_arr * mk_arr + bg_arr[y1:y2, x1:x2] * (1 - mk_arr)
    comp = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8))

    if harmonize_flag:
        full_mask = np.zeros((bg_pil.height, bg_pil.width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = (np.array(mk_s.crop((0, 0, pw, ph))) > 127).astype(np.uint8) * 255
        comp = harmonize(comp, Image.fromarray(full_mask, 'L'))

    # SimOPA score
    bbox = [x1, y1, x2, y2]
    _, score_mask = make_composite(bg_pil, fg_s, mk_s, bbox)
    simopa_score = get_simopa().model.score(comp, score_mask)

    return {
        'status': 'ok',
        'composite': _img_to_base64(comp, 'JPEG'),
        'score_simopa': round(simopa_score, 4),
        'bbox': [bbox[0]/bg_pil.width, bbox[1]/bg_pil.height,
                  bbox[2]/bg_pil.width, bbox[3]/bg_pil.height],
        'scale': round(scale, 1),
        'rotation': round(rotation, 1),
    }


# ===================================================================
#  POST /api/harmonize
# ===================================================================
@app.post('/api/harmonize')
async def api_harmonize(
    composite: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    comp = Image.open(io.BytesIO(await composite.read())).convert('RGB')
    mk = Image.open(io.BytesIO(await mask.read())).convert('L')
    result = harmonize(comp, mk)
    return {
        'status': 'ok',
        'method': 'pctnet' if os.path.exists(PCTNET_PATH) else 'reinhard',
        'image': _img_to_base64(result, 'JPEG'),
    }


# ===================================================================
#  POST /api/mask  — foreground segmentation
# ===================================================================
@app.post('/api/mask')
async def api_mask(
    fg: UploadFile = File(...),
    method: str = Form('sam2'),
):
    fg_np = _read_image(await fg.read())
    if fg_np.shape[-1] == 4:
        mk = fg_np[:, :, 3]
        seg_method = 'alpha'
    elif method == 'sam2':
        mk = auto_mask(fg_np[:, :, :3], prefer_sam=True)
        seg_method = 'sam2'
    else:
        mk = auto_mask(fg_np[:, :, :3], prefer_sam=False)
        seg_method = 'opencv'

    fg_pct = (mk > 127).sum() / mk.size * 100
    mk_pil = Image.fromarray(mk.astype(np.uint8), 'L')
    return {
        'status': 'ok',
        'method': seg_method,
        'mask': _img_to_base64(mk_pil, 'PNG'),
        'mask_width': mk.shape[1],
        'mask_height': mk.shape[0],
        'foreground_percent': round(fg_pct, 1),
    }


# ===================================================================
#  Heatmap drawing (server-side, no Gradio dependency)
# ===================================================================
def _draw_heatmap_server(bg, candidates):
    from scipy.ndimage import gaussian_filter
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
    smap = gaussian_filter(smap, sigma=3.0)
    s_full = np.array(Image.fromarray((smap * 255).astype(np.uint8)).resize((bg_w, bg_h), Image.BILINEAR)).astype(np.float32) / 255.0
    overlay = np.zeros((bg_h, bg_w, 4), dtype=np.uint8)
    overlay[:,:,0] = np.clip(255 * (1.0 - s_full), 0, 255).astype(np.uint8)
    overlay[:,:,1] = np.clip(255 * s_full, 0, 255).astype(np.uint8)
    overlay[:,:,3] = (np.clip(s_full * 0.6 + 0.1, 0, 1) * 255).astype(np.uint8)
    return Image.alpha_composite(bg.convert('RGBA'), Image.fromarray(overlay, 'RGBA')).convert('RGB')


# ===================================================================
#  Entry
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    print(f'Models loaded: simopa')
    topnet = get_topnet()
    print(f'TopNet: {"loaded" if topnet else "not found"}')
    print(f'PCTNet: {"available" if os.path.exists(PCTNET_PATH) else "not found"}')
    uvicorn.run(app, host=args.host, port=args.port)
