"""
TopNet Stage 1 — freeze encoders, train Transformer + Decoder only.

Usage:
  # Original sparse CE
  python train_stage1.py --config configs/stage1.yaml

  # Gaussian + Focal Loss (keypoint style)
  python train_stage1.py --config configs/stage1_focal.yaml

All model/loss/data selection is driven by the YAML config file.
CLI arguments override YAML values.
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


# ======================================================================
#  Config loading
# ======================================================================
def load_config(config_path, cli_overrides=None):
    """Load YAML config and apply CLI overrides."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


# ======================================================================
#  Data loading
# ======================================================================
def get_dataloaders_from_cfg(cfg):
    """Build train + val loaders.

    Training uses the ``data_type`` specified in config (sparse / gaussian).
    Validation ALWAYS uses sparse annotations so that F1/bAcc are comparable
    across experiments.
    """
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

    # Training: config-driven
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


# ======================================================================
#  Metrics
# ======================================================================
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
        # keypoint: target is sparse [B,H,W] with 0/1/255 when used for val
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


# ======================================================================
#  Encoder helpers
# ======================================================================
def _init_4ch_conv1(conv1):
    """Greyscale-initialise the 4th channel from channels 0-2."""
    weight = conv1.weight.data
    for i in range(weight.size(0)):
        weight[i, 3] = (0.299 * weight[i, 0] + 0.587 * weight[i, 1]
                        + 0.114 * weight[i, 2])
    conv1.weight.data = weight


def _copy_imagenet_to_encoder(model, src_state):
    """
    Load ImageNet ResNet18 weights into both encoders.

    Key mappings (ImageNet → our encoder keys):
      conv1.weight          → bg_encoder1.0 / fg_encoder1.0  (3ch → 4ch partial)
      bn1.{w,b,rm,rv,nbt}  → bg_encoder1.1 / fg_encoder1.1
      layer1.*              → bg_encoder2.1.*  / fg_encoder2.1.*
      layer2.*              → bg_encoder4.*    / fg_encoder4.*
      layer3.*              → bg_encoder8.*    / fg_encoder8.*
      layer4.*              → bg_encoder16.*   / fg_encoder16.*  (and 32 for fg)

    conv1 is handled specially: channels 0-2 copied, channel 3 grey-init'd.
    """
    prefix_map = {
        'bg_encoder1':  {'conv1': '0', 'bn1': '1'},
        'bg_encoder2':  {'layer1': '1'},
        'bg_encoder4':  {'layer2': ''},
        'bg_encoder8':  {'layer3': ''},
        'bg_encoder16': {'layer4': ''},
        'fg_encoder1':  {'conv1': '0', 'bn1': '1'},
        'fg_encoder2':  {'layer1': '1'},
        'fg_encoder4':  {'layer2': ''},
        'fg_encoder8':  {'layer3': ''},
        'fg_encoder16': {'layer4': ''},
        'fg_encoder32': {'layer4': ''},
    }
    for enc_name, mappings in prefix_map.items():
        enc = getattr(model, enc_name)
        enc_state = enc.state_dict()
        for src_prefix, dst_prefix in mappings.items():
            # Find all ImageNet keys starting with src_prefix
            prefix_len = len(src_prefix)
            for img_k, img_v in src_state.items():
                if not img_k.startswith(src_prefix):
                    continue
                suffix = img_k[prefix_len:]  # e.g. '.0.conv1.weight' or '.weight'
                if dst_prefix:
                    dst_k = f'{dst_prefix}{suffix}'
                else:
                    dst_k = suffix.lstrip('.')
                if dst_k in enc_state and enc_state[dst_k].shape == img_v.shape:
                    enc_state[dst_k] = img_v.clone()
        enc.load_state_dict(enc_state)

    # conv1 4th channel (both encoders)
    _init_4ch_conv1(model.bg_encoder1[0])
    _init_4ch_conv1(model.fg_encoder1[0])


def _copy_sopa_to_bg_encoder(model, sopa_state):
    """
    Load SOPA weights into bg_encoder.

    SOPA.pth.tar keys are flat, matching the ResNet18 internal structure:
      conv1.*  → bg_encoder1.0.*
      bn1.*    → bg_encoder1.1.*
      layer1.* → bg_encoder2.1.*
      layer2.* → bg_encoder4.*
      layer3.* → bg_encoder8.*
      layer4.* → bg_encoder16.*
      fc.*     → skip (classification head, not needed)
    """
    # Same prefix mapping as ImageNet
    prefix_map = {
        'bg_encoder1':  {'conv1': '0', 'bn1': '1'},
        'bg_encoder2':  {'layer1': '1'},
        'bg_encoder4':  {'layer2': ''},
        'bg_encoder8':  {'layer3': ''},
        'bg_encoder16': {'layer4': ''},
    }
    loaded = 0
    for enc_name, mappings in prefix_map.items():
        enc = getattr(model, enc_name)
        enc_state = enc.state_dict()
        for src_prefix, dst_prefix in mappings.items():
            prefix_len = len(src_prefix)
            for sopa_k, sopa_v in sopa_state.items():
                if not sopa_k.startswith(src_prefix):
                    continue
                suffix = sopa_k[prefix_len:]  # e.g. '.0.conv1.weight'
                if dst_prefix:
                    dst_k = f'{dst_prefix}{suffix}'
                else:
                    dst_k = suffix.lstrip('.')
                if dst_k in enc_state and enc_state[dst_k].shape == sopa_v.shape:
                    enc_state[dst_k] = sopa_v.clone()
                    loaded += 1
        enc.load_state_dict(enc_state)
    print(f'[SOPA] {loaded} keys loaded into bg_encoder.')


def load_encoder_pretrain(model, sopa_path, data_dir):
    """Load pretrained weights for both encoders."""
    import torchvision.models as tv_models

    # ---- ImageNet for fg_encoder + bg_encoder (baseline) ----
    imgnet = tv_models.resnet18(weights='DEFAULT').state_dict()
    _copy_imagenet_to_encoder(model, imgnet)
    print('[Encoder] ImageNet ResNet18 loaded for bg_encoder + fg_encoder.')

    # ---- SOPA for bg_encoder (override if available) ----
    if os.path.exists(sopa_path):
        state = torch.load(sopa_path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        # Some checkpoints wrap in DataParallel
        state = {k.replace('module.', ''): v for k, v in state.items()}
        _copy_sopa_to_bg_encoder(model, state)
    else:
        print(f'[SOPA] {sopa_path} not found — keeping ImageNet weights for bg_encoder.')


def freeze_encoders(model):
    encoder_names = [f'bg_encoder{i}' for i in [1,2,4,8,16]] + \
                    [f'fg_encoder{i}' for i in [1,2,4,8,16,32]]
    for name, param in model.named_parameters():
        if any(name.startswith(en) for en in encoder_names):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'[Stage1] Encoders frozen. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')


# ======================================================================
#  Training / validation
# ======================================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, writer,
                is_gaussian):
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
            # Focal model: 1ch sigmoid → compute BCE loss on sparse target
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
    parser.add_argument('--config', default='configs/stage1_ce.yaml')
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

    # Model
    model = build_model(cfg['model_type']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {cfg["model_type"]} ({n_params:,} params, out={model.out_channels}ch)')

    # Init encoders (ImageNet pretrain for both, SOPA for bg_encoder if available)
    sopa = os.path.join(cfg['data_dir'], 'SOPA.pth.tar')
    load_encoder_pretrain(model, sopa, cfg['data_dir'])
    if cfg.get('freeze_encoders', True):
        freeze_encoders(model)

    # Optimizer & loss
    opt_name = cfg.get('optimizer', 'adam').lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = get_loss(cfg)

    # Logging
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(cfg['log_dir'])

    # Training loop
    best_val_loss = float('inf')
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
            no_improve = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss, 'history': history,
                        'cfg': cfg},
                       os.path.join(cfg['checkpoint_dir'], 'stage1_best.pth'))
            print(f'  -> Best saved (val_loss={val_loss:.4f})')
        else:
            no_improve += 1

        if no_improve >= cfg['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'\nStage 1 done. Best val loss: {best_val_loss:.4f}')
    # Save history + config for plotting / reproducibility
    with open(os.path.join(cfg['log_dir'], 'history.json'), 'w') as f:
        json.dump({'history': history, 'config': cfg}, f, indent=2)
    # Also save a standalone copy of the config
    with open(os.path.join(cfg['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    writer.close()


if __name__ == '__main__':
    main()
