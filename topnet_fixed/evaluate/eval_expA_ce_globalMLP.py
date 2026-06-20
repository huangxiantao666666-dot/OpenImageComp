"""Evaluate expA_ce_globalMLP (fixed attention + buggy-style global MLP)."""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eval_shared import evaluate_model, load_fixed, DataLoader, PlacementDataset

DATA_DIR = './data/data'
WEIGHT  = './checkpoints/expA_ce_globalMLP/stage2_best.pth'
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
NAME    = 'expA_ce_globalMLP'
RESULT_KEY = 'expA_CE_globalMLP'

if not os.path.exists(WEIGHT):
    print(f'[SKIP] {NAME}: checkpoint not found at {WEIGHT}')
    exit(0)

print(f'[{NAME}] Loading model ...')
model, mtype = load_fixed('ObPlaNet_resnet18_globalMLP', WEIGHT, DEVICE)
print(f'[{NAME}] {sum(p.numel() for p in model.parameters()):,} params, {mtype}')

print(f'[{NAME}] Loading test set ...')
ds = PlacementDataset(os.path.join(DATA_DIR, 'test_pair_new.json'),
                       os.path.join(DATA_DIR, 'bg'),
                       os.path.join(DATA_DIR, 'fg'), train=False)
ldr = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

print(f'[{NAME}] Evaluating ...')
r = evaluate_model(model, ldr, DEVICE, model_type=mtype, model_name=NAME)
r['type'] = 'fixed_CE_globalMLP'
os.makedirs('./logs', exist_ok=True)
out = {RESULT_KEY: r}
with open(f'./logs/{NAME}.json', 'w') as f:
    json.dump(out, f, indent=2)
ap_val = r.get('ap', '?')
print(f'[{NAME}] F1={r["f1"]:.4f}  bAcc={r["bAcc"]:.4f}  AP={ap_val}')
print(f'[{NAME}] Saved to logs/{NAME}.json')
