"""
Comprehensive evaluation: buggy TopNet + 3 fixed models on the test set.

Computes per-pixel F1, balanced accuracy, precision, recall for each model
and saves results to ``logs/test_results.json``.
"""

import os, sys, json, time, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.topnet import build_model
from data.dataset import PlacementDataset
from torch.utils.data import DataLoader


# ======================================================================
#  Metrics
# ======================================================================
def evaluate(model, loader, device, is_keypoint=False, model_name='', quick=False):
    """
    Run full evaluation pass.

    Args:
        model:       nn.Module
        loader:      test DataLoader (returns bg, fg, mask, target)
        device:      torch.device
        is_keypoint: True for 1ch heatmap output (threshold at 0.5)
        model_name:  label for tqdm

    Returns:
        dict with 'f1', 'bAcc', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN',
        'inference_time_s', 'params', and 'ap' for keypoint models.
    """
    model.eval()
    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    all_scores = []  # for AP computation (keypoint only)
    all_labels = []
    t0 = time.time()

    for bi, (bg, fg, mask, target) in enumerate(
            tqdm(loader, desc=model_name, leave=False)):
        bg, fg, mask = bg.to(device), fg.to(device), mask.to(device)
        target = target.to(device)

        output = model(bg, fg, mask)
        logits = output[0] if isinstance(output, tuple) else output
        if quick and bi >= 3: break

        if is_keypoint:
            scores = logits.squeeze(1)                    # [B, H, W] probs
            preds = (scores > 0.5).long()
            # Collect raw scores + labels for AP (only on labeled pixels)
            valid = (target != 255)
            for b in range(scores.shape[0]):
                v = valid[b]
                all_scores.append(scores[b][v].flatten())
                all_labels.append(target[b][v].long().flatten())
        else:
            preds = logits.argmax(dim=1)

        valid = (target != 255)
        TP = ((preds == 1) & (target == 1) & valid).sum().item()
        TN = ((preds == 0) & (target == 0) & valid).sum().item()
        FP = ((preds == 1) & (target == 0) & valid).sum().item()
        FN = ((preds == 0) & (target == 1) & valid).sum().item()
        metrics['TP'] += TP
        metrics['TN'] += TN
        metrics['FP'] += FP
        metrics['FN'] += FN

    elapsed = time.time() - t0
    TP, TN, FP, FN = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    bAcc = 0.5 * (TP / max(TP + FN, 1) + TN / max(TN + FP, 1))

    result = {
        'f1': round(f1, 4), 'bAcc': round(bAcc, 4),
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'inference_time_s': round(elapsed, 2),
        'num_samples': len(loader.dataset),
    }

    # AP metrics (keypoint models only)
    if is_keypoint and all_scores:
        ap_data = _compute_ap(all_scores, all_labels)
        result.update(ap_data)

    return result


def _compute_ap(all_scores, all_labels):
    """Compute AP and PR-curve metrics from collected scores and labels.

    Returns dict with 'ap', 'ap50', 'ap75', 'ap90', and per-threshold
    precision/recall/F1 for thresholds 0.5:0.05:0.95.
    """
    import numpy as np
    scores = torch.cat(all_scores).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    order = np.argsort(scores)[::-1]
    labels = labels[order]

    tp = (labels == 1).astype(np.float32)
    fp = (labels == 0).astype(np.float32)
    n_pos = tp.sum()

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    prec_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    rec_curve  = tp_cum / max(n_pos, 1)

    # VOC-style AP: max-precision interpolation at 101 recall levels
    for i in range(len(prec_curve) - 1, 0, -1):
        prec_curve[i - 1] = max(prec_curve[i - 1], prec_curve[i])
    recall_levels = np.linspace(0, 1, 101)
    ap_values = np.interp(recall_levels,
                          np.concatenate([[0.0], rec_curve, [1.0]]),
                          np.concatenate([[1.0], prec_curve, [0.0]]))
    ap = float(np.mean(ap_values))

    # AP at fixed thresholds and range average
    def metrics_at_thresh(t):
        preds_i = (scores >= t).astype(np.int64)
        tp_i = ((preds_i == 1) & (labels == 1)).sum()
        fp_i = ((preds_i == 1) & (labels == 0)).sum()
        fn_i = ((preds_i == 0) & (labels == 1)).sum()
        p_i = tp_i / max(tp_i + fp_i, 1)
        r_i = tp_i / max(tp_i + fn_i, 1)
        f1_i = 2 * p_i * r_i / max(p_i + r_i, 1e-8)
        return f1_i

    result = {'ap': round(ap, 4), 'n_pos': int(n_pos)}
    for t_label, t_val in [('ap50', 0.5), ('ap75', 0.75), ('ap90', 0.9)]:
        result[f'{t_label}_F1'] = round(metrics_at_thresh(t_val), 4)

    # AP over 0.5:0.05:0.95 (average F1 across thresholds)
    thresholds = np.arange(0.5, 0.96, 0.05)
    f1_values = [metrics_at_thresh(t) for t in thresholds]
    result['ap_mean_050_095'] = round(float(np.mean(f1_values)), 4)
    # Per-threshold breakdown
    for t, f1v in zip(thresholds, f1_values):
        result[f'ap@{t:.2f}_F1'] = round(float(f1v), 4)

    return result


# ======================================================================
#  Model builders
# ======================================================================
def load_buggy_topnet(weight_path, device):
    """Load the original buggy TopNet checkpoint (113M params).

    Uses a local copy of the original trainable network code
    (``models/_buggy_network.py``) to avoid path conflicts.
    """
    from models._buggy_network import ObPlaNet_resnet18 as BuggyTopNet
    # pretrained=False: skip SOPA auto-load (checkpoint has all weights)
    model = BuggyTopNet(pretrained=False).to(device)
    state = torch.load(weight_path, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    n = sum(p.numel() for p in model.parameters())
    print(f'  Buggy TopNet: {n:,} params, loaded from {weight_path}')
    return model, n


def load_fixed_model(model_type, weight_path, device):
    """Load one of our fixed TopNet variants."""
    model = build_model(model_type).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    n = sum(p.numel() for p in model.parameters())
    is_kp = (model.out_channels == 1)
    print(f'  {model_type}: {n:,} params (epoch {ckpt["epoch"]}, '
          f'val_loss={ckpt["best_val_loss"]:.4f}), keypoint={is_kp}')
    return model, n, is_kp


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate all TopNet models')
    parser.add_argument('--data_dir', default='./data/data')
    parser.add_argument('--buggy_weight',
                        default='./checkpoints/buggy_best_weight.pth')
    parser.add_argument('--expA_ckpt', default='./checkpoints/expA_ce/stage2_best.pth')
    parser.add_argument('--expA113M_ckpt', default='./checkpoints/expA_ce_113M/stage2_best.pth')
    parser.add_argument('--expADilated_ckpt', default='./checkpoints/expA_ce_dilated/stage2_best.pth')
    parser.add_argument('--expB_ckpt', default='./checkpoints/expB_focal/stage2_best.pth')
    parser.add_argument('--expC_ckpt', default='./checkpoints/expC_focal_full/stage2_best.pth')
    parser.add_argument('--expB113M_ckpt', default='./checkpoints/expB_focal_113M/stage2_best.pth')
    parser.add_argument('--expC113M_ckpt', default='./checkpoints/expC_focal_full_113M/stage2_best.pth')
    parser.add_argument('--quick', action='store_true',
                        help='Only evaluate on first 4 batches (fast check)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ---- Data ----
    test_json = os.path.join(args.data_dir, 'test_pair_new.json')
    bg_dir = os.path.join(args.data_dir, 'bg')
    fg_dir = os.path.join(args.data_dir, 'fg')

    test_ds = PlacementDataset(test_json, bg_dir, fg_dir, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f'Test set: {len(test_ds)} samples, {len(test_loader)} batches')

    # ---- Evaluate all models (with GPU cleanup between) ----
    results = {}

    def _cleanup(model=None):
        if model is not None:
            del model
        import gc
        gc.collect()
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    # Buggy TopNet
    if os.path.exists(args.buggy_weight):
        print('\n--- Buggy TopNet (original, CVPR 2023) ---')
        buggy, buggy_n = load_buggy_topnet(args.buggy_weight, device)
        r = evaluate(buggy, test_loader, device, is_keypoint=False,
                     model_name='Buggy TopNet', quick=args.quick)
        r['params'] = buggy_n
        r['type'] = 'buggy_original'
        results['buggy_original'] = r
        _cleanup(buggy)
    else:
        print(f'\n[Skipping] Buggy TopNet weight not found: {args.buggy_weight}')

    # Exp A
    if os.path.exists(args.expA_ckpt):
        print('\n--- Exp A: Sparse CrossEntropy ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18', args.expA_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=kp, model_name='Exp A', quick=args.quick)
        r['params'] = n
        r['type'] = 'fixed_CE'
        _cleanup(m)
        results['expA_sparse_CE'] = r
    else:
        print(f'\n[Skipping] Exp A not found: {args.expA_ckpt}')

    # Exp A 113M
    if os.path.exists(args.expA113M_ckpt):
        print('\n--- Exp A (113M): Sparse CrossEntropy ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18_113M', args.expA113M_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=False,
                     model_name='Exp A 113M', quick=args.quick)
        r['params'] = n
        _cleanup(m)
        r['type'] = 'fixed_CE_113M'
        results['expA_CE_113M'] = r
    else:
        print(f'\n[Skipping] Exp A 113M not found: {args.expA113M_ckpt}')

    # Exp A Dilated
    if os.path.exists(args.expADilated_ckpt):
        print('\n--- Exp A2: Dilated CrossEntropy (r=3) ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18', args.expADilated_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=False,
                     model_name='Exp A2 dilated', quick=args.quick)
        _cleanup(m)
        r['params'] = n
        r['type'] = 'fixed_CE_dilated'
        results['expA_CE_dilated'] = r
    else:
        print(f'\n[Skipping] Exp A Dilated not found: {args.expADilated_ckpt}')

    # Exp B
    if os.path.exists(args.expB_ckpt):
        print('\n--- Exp B: Partial Focal ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18_keypoint', args.expB_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=kp, model_name='Exp B')
        r['params'] = n
        r['type'] = 'fixed_Focal_partial'
        results['expB_focal_partial'] = r
    else:
        print(f'\n[Skipping] Exp B not found: {args.expB_ckpt}')

    # Exp C
    if os.path.exists(args.expC_ckpt):
        print('\n--- Exp C: Full-supervision Focal ---')
        r = evaluate(m, test_loader, device, is_keypoint=kp, model_name='Exp C')
        r['params'] = n
        r['type'] = 'fixed_Focal_full'
        results['expC_focal_full'] = r
    else:
        print(f'\n[Skipping] Exp C not found: {args.expC_ckpt}')

    # Exp B 113M
    if os.path.exists(args.expB113M_ckpt):
        print('\n--- Exp B (113M): Partial Focal ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18_keypoint_113M',
                                     args.expB113M_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=kp,
                     model_name='Exp B 113M', quick=args.quick)
        r['params'] = n
        r['type'] = 'fixed_Focal_partial_113M'
        results['expB_focal_partial_113M'] = r
    else:
        print(f'\n[Skipping] Exp B 113M not found: {args.expB113M_ckpt}')

    # Exp C 113M
    if os.path.exists(args.expC113M_ckpt):
        print('\n--- Exp C (113M): Full-supervision Focal ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18_keypoint_113M',
                                     args.expC113M_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=kp,
                     model_name='Exp C 113M', quick=args.quick)
        r['params'] = n
        r['type'] = 'fixed_Focal_full_113M'
        results['expC_focal_full_113M'] = r
    else:
        print(f'\n[Skipping] Exp C 113M not found: {args.expC113M_ckpt}')

    # ---- Summary ----
    print('\n' + '=' * 75)
    print('  Evaluation Summary')
    print('=' * 75)
    header = f'  {"Model":<25s} {"F1":>8s} {"bAcc":>8s} {"AP":>8s} {"AP@.5":>8s} {"Time":>8s} {"Params":>10s}'
    print(header)
    print('  ' + '-' * 73)
    for name, r in results.items():
        ap_str = f'{r.get("ap", "-"):>8s}' if isinstance(r.get('ap'), str) else f'{r["ap"]:>8.4f}' if 'ap' in r else f'{"-":>8s}'
        ap50_str = f'{r.get("ap@0.5_F1", "-"):>8s}' if isinstance(r.get('ap@0.5_F1'), str) else f'{r["ap@0.5_F1"]:>8.4f}' if 'ap@0.5_F1' in r else f'{"-":>8s}'
        print(f'  {name:<25s} {r["f1"]:>8.4f} {r["bAcc"]:>8.4f} {ap_str} {ap50_str} '
              f'{r["inference_time_s"]:>7.1f}s {r["params"]:>10,}')

    # Reference from paper
    print(f'\n  Reference (original TopNet paper): F1=0.741  bAcc=0.815')
    print(f'  AP metrics: {"AP" if "ap" in next(iter(results.values())) else "available only for keypoint (1ch) models"}')

    # Save
    os.makedirs('./logs', exist_ok=True)
    save_path = './logs/test_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {save_path}')


if __name__ == '__main__':
    main()
