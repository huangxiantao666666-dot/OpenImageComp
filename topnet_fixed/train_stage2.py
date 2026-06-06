"""
TopNet Stage 2 — unfreeze all, full-model fine-tuning.

Loads stage1 checkpoint (``stage1_best.pth``), unfreezes encoders,
continues training end-to-end at lower LR.  Config-driven same as stage1.

Usage:
  python train_stage2.py --config configs/stage2.yaml
"""

import os, sys, time, json, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.topnet import build_model
from losses import get_loss


# ---- Config loading (same as stage1) ----
def load_config(config_path, cli_overrides=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def get_dataloaders_from_cfg(cfg):
    data_dir = cfg['data_dir']
    train_json = os.path.join(data_dir, 'train_pair_new.json')
    test_json  = os.path.join(data_dir, 'test_pair_new.json')
    bg_dir     = os.path.join(data_dir, 'bg')
    fg_dir     = os.path.join(data_dir, 'fg')
    img_size   = cfg.get('image_size', 256)
    bs         = cfg.get('batch_size', 8)
    nw         = cfg.get('num_workers', 4)

    # Validation: always sparse (no dilation)
    from data.dataset import get_dataloaders as get_sparse
    _, val_loader = get_sparse(train_json, test_json, bg_dir, fg_dir,
                                img_size, bs, nw, label_dilation=0)

    if cfg.get('data_type') == 'gaussian':
        from data.dataset_gaussian import get_dataloaders_gaussian
        sf = cfg.get('gaussian_sigma_factor', 6.0)
        fs = cfg.get('focal_full_supervision', False)
        train_loader, _ = get_dataloaders_gaussian(train_json, test_json,
                                                     bg_dir, fg_dir,
                                                     img_size, bs, nw, sf, fs)
    else:
        dilation = cfg.get('label_dilation', 0)
        train_loader, _ = get_sparse(train_json, test_json, bg_dir, fg_dir,
                                      img_size, bs, nw,
                                      label_dilation=dilation)

    return train_loader, val_loader


# ---- Metrics ----
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum, self.count = 0.0, 0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def compute_metrics(logits, target, ignore_index=255):
    preds = logits.argmax(dim=1) if logits.shape[1] == 2 else \
            (logits > 0.5).long().squeeze(1)
    if logits.shape[1] == 1:
        target_labels = target.long()
        mask = (target_labels != ignore_index)
    else:
        target_labels = target
        mask = (target != ignore_index)
    TP = ((preds == 1) & (target_labels == 1) & mask).sum().item()
    TN = ((preds == 0) & (target_labels == 0) & mask).sum().item()
    FP = ((preds == 1) & (target_labels == 0) & mask).sum().item()
    FN = ((preds == 0) & (target_labels == 1) & mask).sum().item()
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    bAcc = 0.5 * (TP / max(TP + FN, 1) + TN / max(TN + FP, 1))
    return {'f1': f1, 'bAcc': bAcc, 'prec': prec, 'rec': rec,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


# ---- Training / validation ----
def train_epoch(model, loader, criterion, optimizer, device, epoch, writer, is_gaussian):
    model.train()
    loss_m = AverageMeter()
    for batch in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
        if is_gaussian:
            bg, fg, mk, target, valid_mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred = model(bg, fg, mk)
            loss = criterion(pred, target, valid_mask)
        else:
            bg, fg, mk, target = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss = criterion(model(bg, fg, mk), target)
        loss.backward()
        optimizer.step()
        loss_m.update(loss.item(), bg.size(0))
    writer.add_scalar('train/loss', loss_m.avg, epoch)
    return loss_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, is_gaussian):
    model.eval()
    loss_m = AverageMeter()
    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for batch in tqdm(loader, desc='Valid', leave=False):
        # Validation always uses sparse data (4 items), regardless of training type
        bg, fg, mk, target = [b.to(device) for b in batch]
        logits = model(bg, fg, mk)

        if is_gaussian:
            ignore_mask = (target != 255)
            pred_flat = logits.squeeze(1)[ignore_mask]
            tgt_flat = target[ignore_mask].float()
            loss = F.binary_cross_entropy(pred_flat, tgt_flat)
        else:
            loss = criterion(logits, target)

        m = compute_metrics(logits, target, 255)
        loss_m.update(loss.item(), bg.size(0))
        for k in ['TP', 'TN', 'FP', 'FN']:
            metrics[k] += m[k]
    TP, TN, FP, FN = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
    f1 = 2 * (TP/max(TP+FP,1)) * (TP/max(TP+FN,1)) / max((TP/max(TP+FP,1)) + (TP/max(TP+FN,1)), 1e-8)
    bAcc = 0.5 * (TP/max(TP+FN,1) + TN/max(TN+FP,1))
    writer.add_scalar('val/loss', loss_m.avg, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    writer.add_scalar('val/bAcc', bAcc, epoch)
    return loss_m.avg, f1, bAcc


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage2_ce.yaml')
    parser.add_argument('--stage1_ckpt', default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    cfg = load_config(args.config, {
        'data_dir': args.data_dir, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'patience': args.patience,
    })

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    is_gaussian = (cfg.get('data_type') == 'gaussian')
    print(f'Device: {device} | model={cfg["model_type"]} | data={cfg.get("data_type","sparse")} | loss={cfg.get("loss","cross_entropy")}')
    print(f'epochs={cfg["epochs"]} lr={cfg["lr"]} patience={cfg["patience"]}')

    # Data
    train_loader, test_loader = get_dataloaders_from_cfg(cfg)
    print(f'Train: {len(train_loader)} batches, Test: {len(test_loader)} batches')

    # Model — load stage1
    ckpt_path = args.stage1_ckpt or cfg.get('stage1_ckpt', 'checkpoints/stage1_best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Stage1 checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    # Use the same model_type that was used in stage1 (stored in ckpt['cfg'])
    stage1_cfg = ckpt.get('cfg', {})
    model_type = stage1_cfg.get('model_type', cfg['model_type'])
    model = build_model(model_type).to(device)
    model.load_state_dict(ckpt['state_dict'])
    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    print(f'Model: {model_type} ({sum(p.numel() for p in model.parameters()):,} params, '
          f'out={model.out_channels}ch)')
    print(f'Loaded stage1 checkpoint (epoch {ckpt["epoch"]}, val_loss={ckpt["best_val_loss"]:.4f})')

    # Optimizer & loss
    opt_name = cfg.get('optimizer', 'adam').lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                                weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                               weight_decay=cfg['weight_decay'])
    criterion = get_loss(cfg)

    # Logging
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(cfg['log_dir'])

    # Training
    best_val_loss = float('inf')
    best_f1 = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                  device, epoch, writer, is_gaussian)
        val_loss, val_f1, val_bacc = validate(model, test_loader, criterion,
                                               device, epoch, writer, is_gaussian)

        history.append({
            'epoch': epoch, 'lr': cfg['lr'],
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_f1': float(val_f1), 'val_bacc': float(val_bacc),
        })
        print(f'Epoch {epoch:3d} | Train Loss {train_loss:.4f} | '
              f'Val Loss {val_loss:.4f}  F1 {val_f1:.4f}  bAcc {val_bacc:.4f}')

        if val_loss < best_val_loss - cfg.get('early_stop_delta', 1e-4):
            best_val_loss = val_loss
            best_f1 = val_f1
            no_improve = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss, 'best_f1': best_f1,
                        'history': history, 'cfg': cfg},
                       os.path.join(cfg['checkpoint_dir'], 'stage2_best.pth'))
            print(f'  -> Best saved (val_loss={val_loss:.4f}, F1={val_f1:.4f})')
        else:
            no_improve += 1

        if no_improve >= cfg['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'\nStage 2 done. Best val loss: {best_val_loss:.4f} (F1={best_f1:.4f})')
    with open(os.path.join(cfg['log_dir'], 'history.json'), 'w') as f:
        json.dump({'history': history, 'config': cfg}, f, indent=2)
    with open(os.path.join(cfg['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    writer.close()


if __name__ == '__main__':
    main()
