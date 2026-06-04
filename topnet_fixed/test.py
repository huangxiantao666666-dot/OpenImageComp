"""
TopNet evaluation script.

Computes per-pixel F1 and balanced accuracy on the test set.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from models.topnet import ObPlaNet_resnet18
from data.dataset import get_dataloaders
from train import compute_metrics


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute F1 and balanced accuracy on the entire dataset."""
    model.eval()
    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for bg, fg, mask, target in tqdm(loader, desc='Eval'):
        bg, fg = bg.to(device), fg.to(device)
        mask, target = mask.to(device), target.to(device)

        logits = model(bg, fg, mask)
        m = compute_metrics(logits, target)
        for k in ['TP', 'TN', 'FP', 'FN']:
            metrics[k] += m[k]

    TP = metrics['TP']; TN = metrics['TN']
    FP = metrics['FP']; FN = metrics['FN']
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    bAcc = 0.5 * (TP / max(TP + FN, 1) + TN / max(TN + FP, 1))

    print(f'\n{"="*50}')
    print(f'  Evaluation Results')
    print(f'{"="*50}')
    print(f'  TP: {TP:>10,d}    TN: {TN:>10,d}')
    print(f'  FP: {FP:>10,d}    FN: {FN:>10,d}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1:        {f1:.4f}')
    print(f'  bAcc:      {bAcc:.4f}')
    print(f'{"="*50}')
    print(f'\n  Original TopNet (buggy):  F1=0.741  bAcc=0.815')
    print(f'  Fixed TopNet (this):     F1={f1:.3f}  bAcc={bAcc:.3f}')

    return f1, bAcc


def main():
    parser = argparse.ArgumentParser(description='Evaluate TopNet')
    parser.add_argument('--load_path', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--data_dir', default=cfg.DATA_DIR)
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data
    test_json  = os.path.join(args.data_dir, 'test_pair_new.json')
    bg_dir   = os.path.join(args.data_dir, 'bg')
    fg_dir   = os.path.join(args.data_dir, 'fg')
    mask_dir = os.path.join(args.data_dir, 'mask')

    _, test_loader = get_dataloaders(
        None, test_json, bg_dir, fg_dir, mask_dir,
        image_size=cfg.INPUT_SIZE, batch_size=args.batch_size,
        num_workers=cfg.NUM_WORKERS)

    # Model
    model = ObPlaNet_resnet18().to(device)
    ckpt = torch.load(args.load_path, map_location=device)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    print(f'Loaded checkpoint from {args.load_path}')
    print(f'Test batches: {len(test_loader)}')

    evaluate(model, test_loader, device)


if __name__ == '__main__':
    main()
