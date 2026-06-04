# OpenImageComp

Open-source Image Composition Toolkit — a course project combining object placement
assessment, deep harmonization, and interactive Gradio applications.

Built on top of BCMI Lab's image composition research (OPA, TopNet, libcom).

## Project Structure

```
├── placement_app/          # Gradio web application
│   ├── app.py              # Main entry point
│   ├── models/             # SimOPA + TopNet scorer wrappers
│   ├── pipeline/           # Candidate generation, compositing, scoring
│   ├── grad_cam.py         # Grad-CAM model interpretation
│   ├── sam2_demo.py        # SAM2 auto-masking comparison demo
│   └── assets/             # Preset background/foreground images
│
├── topnet_fixed/           # TopNet with fixed Transformer + training pipeline
│   ├── models/topnet.py    # Fixed ObPlaNet (dual-encoder + Transformer + UNet)
│   ├── losses.py           # CrossEntropy + FocalLoss
│   ├── data/               # Sparse + Gaussian-heatmap dataset loaders
│   ├── configs/            # 6 YAML configs for 3 ablation experiments
│   ├── train_stage1.py     # Stage 1: freeze encoders, train Transformer+Decoder
│   ├── train_stage2.py     # Stage 2: unfreeze all, full-model fine-tuning
│   ├── test.py             # Evaluation script (F1 + balanced accuracy)
│   └── sh_scripts/         # Bash run scripts
│
├── OPA/                    # Original OPA reference code
├── libcom-main/            # libcom reference toolbox
├── TopNet-Object-Placement-main/  # Original TopNet reference code
├── papers/                 # Reference papers (PDF)
├── Design.md               # Technical analysis of all three projects
├── CodeDesign.md           # Detailed code walkthrough
└── 方向A实施方案.md         # Implementation plan (Chinese)
```

## Quick Start — Gradio App

```bash
cd placement_app
pip install -r requirements.txt
python app.py          # → http://127.0.0.1:7860
```

**Placement methods:**
- **TopNet** (CVPR 2023): single forward pass → 256×256 heatmap → top-K + SimOPA fine-scoring
- **Grid + SimOPA**: enumerate grid → composite → score each
- **Manual**: click-to-place on background → SimOPA scoring

**Harmonization:** PCTNet (ViT-based) or Reinhard colour transfer

**Auto-masking:** SAM2.1 / OpenCV / alpha channel / manual upload

## Training — TopNet (Fixed Transformer)

### Data Preparation

Download from [Baidu Cloud](https://pan.baidu.com/s/10JBpXBMZybEl5FTqBlq-hQ) (code: `4zf9`)
or [Dropbox](https://www.dropbox.com/scl/fi/c05wk038piy224sba6jpi/data.rar).
Also download [SOPA.pth.tar](https://pan.baidu.com/s/1hQGm3ryRONRZpNpU66SJZA) (code: `1x3n`).

Place under `topnet_fixed/data/data/`:

```
data/data/
├── bg/                         # background images
├── fg/                         # foreground images
│   └── foreground/             # source fg + mask_{id}.jpg
├── train_pair_new.json         # training annotations
├── test_pair_new.json          # test annotations
└── SOPA.pth.tar                # SOPA encoder weights
```

### Three Ablation Experiments

| Exp | Config | Model | Loss | Description |
|-----|--------|-------|------|-------------|
| A | `stage1_ce.yaml` | 2ch logits | CrossEntropy(ignore=255) | Sparse baseline |
| B | `stage1_focal.yaml` | 1ch heatmap | Focal (partial, α=2, β=4) | Gaussian peaks, pos+neg only |
| C | `stage1_focal_full.yaml` | 1ch heatmap | Focal (full, α=2, β=4) | Gaussian peaks, all pixels supervised |

### Run Training

```bash
cd topnet_fixed

# Single experiment
bash sh_scripts/expA_ce.sh          # Exp A: stage1 → stage2
bash sh_scripts/expB_focal.sh       # Exp B: stage1 → stage2
bash sh_scripts/expC_focal_full.sh  # Exp C: stage1 → stage2

# All three sequentially
bash sh_scripts/run_all.sh

# Or single stage
python train_stage1.py --config configs/stage1_ce.yaml --device cuda
python train_stage2.py --config configs/stage2_ce.yaml --device cuda
```

### Test

```bash
python test.py --load_path checkpoints/expA_ce/stage2_best.pth
```

## Model Architecture

### SimOPA (Object Placement Assessment)
- 4-channel ResNet18 (RGB + mask) → GAP → FC(512, 2)
- 11.2M params, scores a single composite image

### TopNet / ObPlaNet (Fixed, CVPR 2023)
- Dual 4-channel ResNet18 encoders (SOPA-pretrained bg, ImageNet-pretrained fg)
- 4-layer Transformer with per-token self-attention (8 heads, 1024-dim)
- UNet decoder with background skip connections
- FIXED: LayerNorm on feature dim (1024) not spatial (8)
- FIXED: MHA uses batch_first=True for correct cross-position attention
- 79.4M params (113M original)
- Output: `[B, 2, 256, 256]` (binary) or `[B, 1, 256, 256]` (keypoint heatmap)

## References

- OPA: [Object Placement Assessment Dataset](https://arxiv.org/abs/2107.01889) (arXiv 2021)
- TopNet: [Transformer-based Object Placement Network](https://github.com/bcmi/TopNet-Object-Placement) (CVPR 2023)
- SOPA / FOPA: [BCMI Lab](https://github.com/bcmi/libcom)
- SAM2: [Segment Anything 2](https://github.com/facebookresearch/sam2) (Meta)
- CenterNet: [Objects as Points](https://arxiv.org/abs/1904.07850) (Zhou et al., 2019)
- PCTNet / Reinhard: from [libcom](https://github.com/bcmi/libcom) harmonization toolkit

## License

MIT (aligned with OPA/libcom reference code)
