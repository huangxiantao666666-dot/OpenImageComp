"""
Shared evaluation utilities.  Each per-model script in this directory
imports this module, loads the specified checkpoint, runs evaluation on
the test set, and saves a per-model JSON result to ``logs/``.
"""

import os, sys, time, json, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.topnet import build_model
from data.dataset import PlacementDataset
from torch.utils.data import DataLoader


# ======================================================================
#  Metrics
# ======================================================================
def compute_ap(all_scores, all_labels):
    """VOC-style interpolated AP + per-threshold F1 at 0.5:0.05:0.95."""
    scores = torch.cat(all_scores).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    order = np.argsort(scores)[::-1]
    labels = labels[order]

    tp = (labels == 1).astype(np.float32)
    fp = (labels == 0).astype(np.float32)
    n_pos = tp.sum()

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    prec_c = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    rec_c  = tp_cum / max(n_pos, 1)

    for i in range(len(prec_c) - 1, 0, -1):
        prec_c[i - 1] = max(prec_c[i - 1], prec_c[i])
    rl = np.linspace(0, 1, 101)
    ap = float(np.mean(np.interp(rl,
        np.concatenate([[0.0], rec_c, [1.0]]),
        np.concatenate([[1.0], prec_c, [0.0]]))))

    def f1_at(t):
        p = ((scores >= t) & (labels == 1)).sum()
        fp_v = ((scores >= t) & (labels == 0)).sum()
        fn_v = ((scores < t) & (labels == 1)).sum()
        prec = p / max(p + fp_v, 1)
        rec  = p / max(p + fn_v, 1)
        return round(float(2 * prec * rec / max(prec + rec, 1e-8)), 4)

    result = {'ap': round(ap, 4), 'n_pos': int(n_pos)}
    for tl, tv in [('ap50_F1', 0.5), ('ap75_F1', 0.75), ('ap90_F1', 0.9)]:
        result[tl] = f1_at(tv)
    thresh = np.arange(0.5, 0.96, 0.05)
    f1s = [f1_at(t) for t in thresh]
    result['ap_mean_050_095'] = round(float(np.mean(f1s)), 4)
    for t, v in zip(thresh, f1s):
        result[f'ap@{t:.2f}_F1'] = v
    return result


# ======================================================================
#  Evaluate one model
# ======================================================================
def evaluate_model(model, loader, device, model_type='2ch', model_name=''):
    """Run evaluation, return metrics dict."""
    model.eval()
    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    all_scores, all_labels = [], []
    t0 = time.time()
    is_kp = (model_type == '1ch')

    for bg, fg, mk, target in tqdm(loader, desc=model_name, leave=False):
        bg, fg, mk = bg.to(device), fg.to(device), mk.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(bg, fg, mk)
        logits = output[0] if isinstance(output, tuple) else output

        valid = (target != 255)
        if is_kp:
            scores = logits.squeeze(1)
            preds = (scores > 0.5).long()
        else:
            # 2ch: use softmax class-1 prob as score for AP
            scores = F.softmax(logits, dim=1)[:, 1]  # [B, H, W]
            preds = logits.argmax(dim=1)
        for b in range(scores.shape[0]):
            v = valid[b]
            all_scores.append(scores[b][v].flatten())
            all_labels.append(target[b][v].long().flatten())

        valid = (target != 255)
        metrics['TP'] += ((preds == 1) & (target == 1) & valid).sum().item()
        metrics['TN'] += ((preds == 0) & (target == 0) & valid).sum().item()
        metrics['FP'] += ((preds == 1) & (target == 0) & valid).sum().item()
        metrics['FN'] += ((preds == 0) & (target == 1) & valid).sum().item()

    elapsed = time.time() - t0
    TP, TN, FP, FN = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    result = {
        'f1': round(2 * prec * rec / max(prec + rec, 1e-8), 4),
        'bAcc': round(0.5 * (TP / max(TP + FN, 1) + TN / max(TN + FP, 1)), 4),
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'inference_time_s': round(elapsed, 2),
        'num_samples': len(loader.dataset),
    }
    if all_scores:
        result.update(compute_ap(all_scores, all_labels))
    result['params'] = sum(p.numel() for p in model.parameters())
    return result


# ======================================================================
#  Model loaders
# ======================================================================
def load_buggy(weight_path, device):
    from models._buggy_network import ObPlaNet_resnet18 as Buggy
    m = Buggy(pretrained=False).to(device)
    s = torch.load(weight_path, map_location=device)
    s = {k.replace('module.', ''): v for k, v in s.items()}
    m.load_state_dict(s, strict=True)
    return m, '2ch'

def load_fixed(model_type, weight_path, device):
    m = build_model(model_type).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    m.load_state_dict(ckpt['state_dict'])
    return m, ('1ch' if m.out_channels == 1 else '2ch')
