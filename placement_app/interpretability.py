"""
Model interpretability for SimOPA — occlusion, saliency, feature visualization.

All methods take a 4ch tensor [1, 4, 256, 256] (RGB + mask) and produce
visualizations as PIL Images or numpy arrays.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw


# ======================================================================
#  Saliency Map  (input gradients)
# ======================================================================
def saliency_map(model, img_cat, class_idx=1):
    """
    Compute per-pixel saliency: ‖∂score/∂input‖.

    Returns:
        numpy [256, 256] float, normalized to [0, 1].
    """
    x = img_cat.clone().requires_grad_(True)
    logits = model(x)
    score = logits[0, class_idx]
    model.zero_grad()
    score.backward()

    sal = x.grad.abs().max(dim=1)[0]          # [1, 256, 256] → max over channels
    sal = sal[0].detach().cpu().numpy()
    return _normalize(sal)


# ======================================================================
#  Occlusion Map
# ======================================================================
def occlusion_map(model, img_cat, mask_tensor=None, window=32, stride=16,
                  class_idx=1):
    """
    Slide a grey (128) window over the image, record score drop at each
    position.  Produces a coarse importance map.

    Args:
        img_cat:  [1, 4, 256, 256]
        window:   occlusion patch size.
        stride:   step between occlusion positions.
        class_idx: 0=unreasonable, 1=reasonable.

    Returns:
        numpy [H, W] normalized to [0, 1].
    """
    _, C, H, W = img_cat.shape
    baseline_score = _get_score(model, img_cat, class_idx)

    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            # Create occluded copy
            occluded = img_cat.clone()
            # Grey fill for RGB channels, keep mask channel as-is
            occluded[:, :3, y:y+window, x:x+window] = 0.5
            # Also occlude the mask channel in the same region
            if C == 4:
                occluded[:, 3, y:y+window, x:x+window] = 0.0

            score = _get_score(model, occluded, class_idx)
            delta = baseline_score - score          # positive = score drops (important)
            heat[y:y+window, x:x+window] += delta
            count[y:y+window, x:x+window] += 1

    heat /= (count + 1e-8)
    return _normalize(heat)


# ======================================================================
#  Feature Visualization  (per-layer activations)
# ======================================================================
def feature_maps(model, img_cat, layer_name='layer2'):
    """
    Extract and visualize feature maps from a specific ResNet layer.

    Args:
        model:    SimOPA model.
        img_cat:  [1, 4, 256, 256].
        layer_name: 'layer1' | 'layer2' | 'layer3' | 'layer4'.

    Returns:
        PIL Image with an 8×8 grid of the first 64 feature channels,
        averaged over spatial dimensions for display.
    """
    activations = {}

    # Map layer names to backbone indices
    layer_idx = {'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}
    target = model.backbone[layer_idx.get(layer_name, 7)]

    def hook(module, inp, out):
        activations['out'] = out.detach()

    h = target.register_forward_hook(hook)
    model(img_cat)
    h.remove()

    feats = activations['out'][0]                # [C, H, W]
    C = min(feats.shape[0], 64)                   # show at most 64 channels
    feats = feats[:C]

    # Normalize each channel independently
    feats = feats - feats.amin(dim=(1, 2), keepdim=True)
    denom = feats.amax(dim=(1, 2), keepdim=True) + 1e-8
    feats = feats / denom

    # Arrange in 8×8 grid
    grid_size = 8
    h_in, w_in = feats.shape[1], feats.shape[2]
    canvas = np.zeros((grid_size * h_in, grid_size * w_in), dtype=np.uint8)

    for i in range(min(C, grid_size * grid_size)):
        row, col = i // grid_size, i % grid_size
        canvas[row*h_in:(row+1)*h_in, col*w_in:(col+1)*w_in] = \
            (feats[i].cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(canvas)


# ======================================================================
#  All-in-one: generate all 4 interpretability outputs
# ======================================================================
def generate_all(model, composite, mask, class_idx=1):
    """
    Generate all 4 interpretability visualizations from a PIL composite + mask.

    Returns:
        dict with keys 'gradcam', 'saliency', 'occlusion', 'features_layer2',
        'features_layer4'.  Values are PIL Images.
    """
    import torchvision.transforms as T
    to_t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    img_t = to_t(composite.convert('RGB'))
    msk_t = to_t(mask.convert('L'))
    cat = torch.cat([img_t, msk_t], dim=0).unsqueeze(0)

    results = {}

    # Grad-CAM
    from grad_cam import GradCAM
    try:
        gc = GradCAM(model)
        results['gradcam'] = gc.overlay(cat, class_idx=class_idx, alpha=0.45)
        gc.remove()  # clean up hook so it doesn't interfere later
    except Exception as e:
        print(f'[Interp] GradCAM failed: {e}')
        results['gradcam'] = None

    # Saliency
    try:
        sal = saliency_map(model, cat, class_idx)
        results['saliency'] = _overlay_heat(cat, sal)
    except Exception as e:
        print(f'[Interp] Saliency failed: {e}')
        results['saliency'] = None

    # Occlusion
    try:
        occ = occlusion_map(model, cat, window=40, stride=40, class_idx=class_idx)
        results['occlusion'] = _overlay_heat(cat, occ)
    except Exception as e:
        print(f'[Interp] Occlusion failed: {e}')
        results['occlusion'] = None

    # Feature maps
    try:
        for layer in ['layer2', 'layer4']:
            results[f'features_{layer}'] = feature_maps(model, cat, layer)
    except Exception as e:
        print(f'[Interp] FeatureViz failed: {e}')
        results['features_layer2'] = None
        results['features_layer4'] = None

    return results


# ======================================================================
#  Helpers
# ======================================================================
@torch.no_grad()
def _get_score(model, img_cat, class_idx):
    logits = model(img_cat)
    return F.softmax(logits, dim=-1)[0, class_idx].item()


def _normalize(arr):
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / max(mx - mn, 1e-8)


def _apply_jet(grey):
    """Jet colormap — no matplotlib dependency."""
    c = np.zeros((*grey.shape, 3), dtype=np.float32)
    c[..., 0] = np.clip(np.minimum(4 * grey - 1.5, -4 * grey + 4.5), 0, 1)
    c[..., 1] = np.clip(np.minimum(4 * grey - 0.5, -4 * grey + 3.5), 0, 1)
    c[..., 2] = np.clip(np.minimum(4 * grey + 0.5, -4 * grey + 2.5), 0, 1)
    return c


def _overlay_heat(img_cat, heat_1d, alpha=0.5):
    """Overlay heatmap on RGB channels of the 4ch input."""
    rgb = img_cat[0, :3].detach().cpu().numpy().transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-8)
    hmap = _apply_jet(heat_1d)
    blended = rgb * (1 - alpha) + hmap * alpha
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))
