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
from data.dataset import get_dataloaders


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
        'inference_time_s', 'params'
    """
    model.eval()
    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    t0 = time.time()

    for bi, (bg, fg, mask, target) in enumerate(
            tqdm(loader, desc=model_name, leave=False)):
        bg, fg, mask = bg.to(device), fg.to(device), mask.to(device)
        target = target.to(device)

        output = model(bg, fg, mask)
        logits = output[0] if isinstance(output, tuple) else output
        if quick and bi >= 3: break

        if is_keypoint:
            # [B, 1, H, W] → threshold at 0.5 → [B, H, W]
            preds = (logits.squeeze(1) > 0.5).long()
        else:
            # [B, 2, H, W] → argmax
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

    return {
        'f1': round(f1, 4), 'bAcc': round(bAcc, 4),
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'inference_time_s': round(elapsed, 2),
        'num_samples': len(loader.dataset),
    }


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
                        default='../TopNet-Object-Placement-main/best_weight.pth')
    parser.add_argument('--expA_ckpt', default='./checkpoints/expA_ce/stage2_best.pth')
    parser.add_argument('--expB_ckpt', default='./checkpoints/expB_focal/stage2_best.pth')
    parser.add_argument('--expC_ckpt', default='./checkpoints/expC_focal_full/stage2_best.pth')
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

    _, test_loader = get_dataloaders(
        None, test_json, bg_dir, fg_dir,
        image_size=256, batch_size=args.batch_size, num_workers=4)
    print(f'Test set: {len(test_loader.dataset)} samples, '
          f'{len(test_loader)} batches')

    # ---- Evaluate all models ----
    results = {}

    # Buggy TopNet
    if os.path.exists(args.buggy_weight):
        print('\n--- Buggy TopNet (original, CVPR 2023) ---')
        buggy, buggy_n = load_buggy_topnet(args.buggy_weight, device)
        r = evaluate(buggy, test_loader, device, is_keypoint=False,
                     model_name='Buggy TopNet', quick=args.quick)
        r['params'] = buggy_n
        r['type'] = 'buggy_original'
        results['buggy_original'] = r
    else:
        print(f'\n[Skipping] Buggy TopNet weight not found: {args.buggy_weight}')

    # Exp A
    if os.path.exists(args.expA_ckpt):
        print('\n--- Exp A: Sparse CrossEntropy ---')
        m, n, kp = load_fixed_model('ObPlaNet_resnet18', args.expA_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=kp, model_name='Exp A', quick=args.quick)
        r['params'] = n
        r['type'] = 'fixed_CE'
        results['expA_sparse_CE'] = r
    else:
        print(f'\n[Skipping] Exp A not found: {args.expA_ckpt}')

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
        m, n, kp = load_fixed_model('ObPlaNet_resnet18_keypoint', args.expC_ckpt, device)
        r = evaluate(m, test_loader, device, is_keypoint=kp, model_name='Exp C')
        r['params'] = n
        r['type'] = 'fixed_Focal_full'
        results['expC_focal_full'] = r
    else:
        print(f'\n[Skipping] Exp C not found: {args.expC_ckpt}')

    # ---- Summary ----
    print('\n' + '=' * 75)
    print('  Evaluation Summary')
    print('=' * 75)
    header = f'  {"Model":<25s} {"F1":>8s} {"bAcc":>8s} {"Prec":>8s} {"Recall":>8s} {"Time":>8s} {"Params":>10s}'
    print(header)
    print('  ' + '-' * 73)
    for name, r in results.items():
        print(f'  {name:<25s} {r["f1"]:>8.4f} {r["bAcc"]:>8.4f} '
              f'{r["precision"]:>8.4f} {r["recall"]:>8.4f} '
              f'{r["inference_time_s"]:>7.1f}s {r["params"]:>10,}')

    # Reference from paper
    print(f'\n  Reference (original TopNet paper): F1=0.741  bAcc=0.815')

    # Save
    os.makedirs('./logs', exist_ok=True)
    save_path = './logs/test_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {save_path}')


if __name__ == '__main__':
    main()
