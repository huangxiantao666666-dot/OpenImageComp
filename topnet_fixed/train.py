"""
TopNet training script (fixed Transformer version).

Train ObPlaNet_resnet18 from scratch on the SOPA/OPA placement dataset.
"""

import os
import sys
import time
import datetime
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from models.topnet import ObPlaNet_resnet18
from data.dataset import get_dataloaders


# ======================================================================
#  Metrics
# ======================================================================
def compute_metrics(logits, target, ignore_index=255):
    """
    Compute per-pixel F1 and balanced accuracy.

    Args:
        logits:  [B, 2, H, W]  model output.
        target:  [B, H, W]     ground truth (0=neg, 1=pos, 255=ignore).

    Returns:
        dict with 'f1', 'bAcc', 'TP', 'TN', 'FP', 'FN'.
    """
    preds = logits.argmax(dim=1)                           # [B, H, W]
    mask = (target != ignore_index)

    TP = ((preds == 1) & (target == 1) & mask).sum().item()
    TN = ((preds == 0) & (target == 0) & mask).sum().item()
    FP = ((preds == 1) & (target == 0) & mask).sum().item()
    FN = ((preds == 0) & (target == 1) & mask).sum().item()

    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    TPR = TP / max(TP + FN, 1)      # sensitivity / recall
    TNR = TN / max(TN + FP, 1)      # specificity
    b_acc = 0.5 * (TPR + TNR)

    return {
        'f1': f1,
        'bAcc': b_acc,
        'precision': precision,
        'recall': recall,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
    }


# ======================================================================
#  Training helpers
# ======================================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def load_sopa_encoder(model, sopa_path):
    """
    Load SOPA-pretrained 4ch ResNet18 weights into the background encoder.

    The SOPA checkpoint is expected to contain a state_dict matching the
    4ch ResNet18 architecture used by the bg_encoder.
    """
    if not os.path.exists(sopa_path):
        print(f'[WARNING] SOPA weight not found at {sopa_path}. '
              f'Training bg_encoder from scratch.')
        return

    state = torch.load(sopa_path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}

    # Build a dict of bg_encoder params
    bg_params = {}
    for name in ['bg_encoder1', 'bg_encoder2', 'bg_encoder4',
                 'bg_encoder8', 'bg_encoder16']:
        for k, v in getattr(model, name).state_dict().items():
            bg_params[f'{name}.{k}'] = v

    # Try to match SOPA keys to bg_encoder keys
    loaded = 0
    for k, v in bg_params.items():
        if k in state and state[k].shape == v.shape:
            bg_params[k] = state[k]
            loaded += 1

    model.load_state_dict(bg_params, strict=False)
    print(f'[SOPA] Loaded {loaded} matching keys into bg_encoder.')


def init_fg_encoder(model):
    """
    Initialise the 4th channel of the foreground encoder's conv1 using the
    greyscale formula (same as the original TopNet).
    """
    conv1 = model.fg_encoder1[0]  # nn.Conv2d(4, 64, 7, ...)
    weight = conv1.weight.data                             # [64, 4, 7, 7]
    # Copy channels 0-2 from ImageNet pretrained (already loaded if pretrained)
    # Initialise channel 3 as greyscale average
    for i in range(weight.size(0)):
        weight[i, 3] = (0.299 * weight[i, 0] +
                        0.587 * weight[i, 1] +
                        0.114 * weight[i, 2])
    conv1.weight.data = weight
    print('[FG-Encoder] Greyscale-init applied to 4th conv1 channel.')


# ======================================================================
#  Epoch loops
# ======================================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    loss_meter = AverageMeter()
    metrics_accum = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    loop = tqdm(loader, desc=f'Epoch {epoch:3d}', leave=False)
    for batch_idx, (bg, fg, mask, target) in enumerate(loop):
        bg, fg = bg.to(device), fg.to(device)
        mask, target = mask.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(bg, fg, mask)                       # [B, 2, H, W]
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), bg.size(0))

        # Accumulate metrics
        m = compute_metrics(logits.detach(), target)
        for k in ['TP', 'TN', 'FP', 'FN']:
            metrics_accum[k] += m[k]

        if batch_idx % cfg.PRINT_FREQ == 0:
            loop.set_postfix(loss=loss_meter.avg,
                             acc=f'{m["bAcc"]:.3f}',
                             f1=f'{m["f1"]:.3f}')

    # Epoch-level metrics
    TP = metrics_accum['TP']; TN = metrics_accum['TN']
    FP = metrics_accum['FP']; FN = metrics_accum['FN']
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    bAcc = 0.5 * (TP / max(TP + FN, 1) + TN / max(TN + FP, 1))

    writer.add_scalar('train/loss', loss_meter.avg, epoch)
    writer.add_scalar('train/f1', f1, epoch)
    writer.add_scalar('train/bAcc', bAcc, epoch)

    return loss_meter.avg, f1, bAcc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    loss_meter = AverageMeter()
    metrics_accum = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for bg, fg, mask, target in tqdm(loader, desc='Valid', leave=False):
        bg, fg = bg.to(device), fg.to(device)
        mask, target = mask.to(device), target.to(device)

        logits = model(bg, fg, mask)
        loss = criterion(logits, target)

        loss_meter.update(loss.item(), bg.size(0))
        m = compute_metrics(logits, target)
        for k in ['TP', 'TN', 'FP', 'FN']:
            metrics_accum[k] += m[k]

    TP = metrics_accum['TP']; TN = metrics_accum['TN']
    FP = metrics_accum['FP']; FN = metrics_accum['FN']
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    bAcc = 0.5 * (TP / max(TP + FN, 1) + TN / max(TN + FP, 1))

    writer.add_scalar('val/loss', loss_meter.avg, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    writer.add_scalar('val/bAcc', bAcc, epoch)

    return loss_meter.avg, f1, bAcc


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Train TopNet (fixed)')
    parser.add_argument('--data_dir', default=cfg.DATA_DIR)
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cfg.LR)
    parser.add_argument('--weight_decay', type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default=None, help='Path to checkpoint')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}')

    # ---- Data ----
    train_json = os.path.join(args.data_dir, 'train_pair_new.json')
    test_json  = os.path.join(args.data_dir, 'test_pair_new.json')
    bg_dir   = os.path.join(args.data_dir, 'bg')
    fg_dir   = os.path.join(args.data_dir, 'fg')

    if not os.path.exists(train_json):
        raise FileNotFoundError(
            f'{train_json} not found. Download the dataset from '
            f'Baidu Cloud (code: 4zf9) or Dropbox. See README.')

    train_loader, test_loader = get_dataloaders(
        train_json, test_json, bg_dir, fg_dir,
        image_size=cfg.INPUT_SIZE, batch_size=args.batch_size,
        num_workers=cfg.NUM_WORKERS)
    print(f'Train batches: {len(train_loader)}, '
          f'Test batches: {len(test_loader)}')

    # ---- Model ----
    model = ObPlaNet_resnet18().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    # Initialise bg encoder with SOPA weights
    sopa_path = os.path.join(args.data_dir, 'SOPA.pth.tar')
    load_sopa_encoder(model, sopa_path)
    init_fg_encoder(model)

    # ---- Resume ----
    start_epoch = 1
    best_val_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 1)
        best_val_f1 = ckpt.get('best_f1', 0.0)
        print(f'Resumed from {args.resume} (epoch {start_epoch})')

    # ---- Optimizer & Loss ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)

    # ---- Logging ----
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(cfg.LOG_DIR) if cfg.LOG_TO_TENSORBOARD else None

    # ---- Training loop ----
    t_start = time.time()
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        # Adjust LR
        current_lr = args.lr * (cfg.LR_DECAY ** (epoch // cfg.DECAY_EVERY))
        for g in optimizer.param_groups:
            g['lr'] = current_lr

        train_loss, train_f1, train_bacc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer)

        val_loss, val_f1, val_bacc = validate(
            model, test_loader, criterion, device, epoch, writer)

        history.append({
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_loss, 'train_f1': train_f1, 'train_bacc': train_bacc,
            'val_loss': val_loss, 'val_f1': val_f1, 'val_bacc': val_bacc,
        })

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss {train_loss:.4f} F1 {train_f1:.4f} bAcc {train_bacc:.4f} | '
              f'Val Loss {val_loss:.4f} F1 {val_f1:.4f} bAcc {val_bacc:.4f} | '
              f'LR {current_lr:.2e}')

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'history': history,
            }, os.path.join(cfg.CHECKPOINT_DIR, 'best.pth'))
            print(f'  -> Best model saved (F1={val_f1:.4f})')

        # Periodic save
        if epoch % cfg.SAVE_FREQ == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'history': history,
            }, os.path.join(cfg.CHECKPOINT_DIR, f'model_epoch{epoch:03d}.pth'))

    # ---- Final ----
    t_total = time.time() - t_start
    print(f'\nTraining complete in {datetime.timedelta(seconds=int(t_total))}')
    print(f'Best val F1: {best_val_f1:.4f}')

    # Save history as JSON
    with open(os.path.join(cfg.LOG_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
