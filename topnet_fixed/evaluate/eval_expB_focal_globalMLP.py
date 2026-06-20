"""Evaluate expB_focal_globalMLP (fixed attention + buggy-style global MLP, partial Focal)."""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eval_shared import evaluate_model, load_fixed, DataLoader, PlacementDataset

DATA_DIR = './data/data'
WEIGHT  = './checkpoints/expB_focal_globalMLP/stage2_best.pth'
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
NAME    = 'expB_focal_globalMLP'
RESULT_KEY = 'expB_focal_globalMLP'

if not os.path.exists(WEIGHT):
    print(f'[SKIP] {NAME}: checkpoint not found at {WEIGHT}')
    exit(0)

print(f'[{NAME}] Loading model ...')
model, mtype = load_fixed('ObPlaNet_resnet18_keypoint_globalMLP', WEIGHT, DEVICE)
print(f'[{NAME}] {sum(p.numel() for p in model.parameters()):,} params, {mtype}')

print(f'[{NAME}] Loading test set ...')
ds = PlacementDataset(os.path.join(DATA_DIR, 'test_pair_new.json'),
                       os.path.join(DATA_DIR, 'bg'),
                       os.path.join(DATA_DIR, 'fg'), train=False)
ldr = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

print(f'[{NAME}] Evaluating ...')
r = evaluate_model(model, ldr, DEVICE, model_type=mtype, model_name=NAME)
r['type'] = 'fixed_Focal_partial_globalMLP'
os.makedirs('./logs', exist_ok=True)
out = {RESULT_KEY: r}
with open(f'./logs/{NAME}.json', 'w') as f:
    json.dump(out, f, indent=2)
ap_val = r.get('ap', '?')
f1_val = r['f1']
bacc_val = r['bAcc']
print('[{0}] F1={1:.4f}  bAcc={2:.4f}  AP={3}'.format(NAME, f1_val, bacc_val, ap_val))
print(f'[{NAME}] Saved to logs/{NAME}.json')
