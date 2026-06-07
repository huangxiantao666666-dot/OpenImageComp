# OpenImageComp

Open-source Image Composition Toolkit — a course project combining object placement
assessment, deep harmonization, interactive Gradio applications, and an Android app.

Built on top of BCMI Lab's image composition research (OPA, TopNet, libcom).

## Project Structure

```
├── placement_app/              # Interactive application (Gradio + REST API)
│   ├── app.py                  # Gradio web app (3 placement modes)
│   ├── server.py               # FastAPI server for Android app
│   ├── sam2_demo.py            # SAM2 auto-masking comparison demo
│   ├── grad_cam.py             # Grad-CAM model interpretation
│   ├── models/                 # SimOPA + TopNet + PCTNet wrappers
│   ├── pipeline/               # Compositing, scoring, harmonization, auto-masking
│   └── assets/                 # 8 background + 8 foreground presets (from OPA dataset)
│
├── topnet_fixed/               # TopNet with fixed Transformer + training pipeline
│   ├── models/topnet.py        # Fixed ObPlaNet (79M/113M, 2ch/1ch variants)
│   ├── losses.py               # CrossEntropy + FocalLoss (alpha=2, beta=4)
│   ├── data/                   # Sparse CE + Gaussian-heatmap + Dilation dataset loaders
│   ├── configs/                # 10 YAML configs for ablation experiments
│   ├── train_stage1.py         # Stage 1: freeze encoders, train Transformer+Decoder
│   ├── train_stage2.py         # Stage 2: unfreeze all, full-model fine-tuning
│   ├── test_exp.py             # Multi-model evaluation script
│   └── sh_scripts/             # Bash run scripts
│
├── OPA/                        # Original OPA reference code
├── libcom-main/                # libcom reference toolbox
├── TopNet-Object-Placement-main/  # Original TopNet (buggy) reference code
├── papers/                     # Reference papers (PDF)
├── Design.md                   # Technical analysis
├── CodeDesign.md               # Detailed code walkthrough
└── ImageCompApp/               # Android app (Jetpack Compose + Retrofit)
```

## Quick Start — Gradio App

```bash
cd placement_app
pip install -r requirements.txt
python app.py          # → http://127.0.0.1:7860
```

**Three placement modes:**
- **Auto Search** (TopNet heatmap → SimOPA fine-scoring, or Grid enumeration)
- **Click-to-Place** (tap on background → foreground composited at tap point)
- **Manual Controls** (scale slider, rotation, harmonization toggle)

**Features:**
- SAM2.1 / OpenCV / alpha-channel foreground segmentation
- PCTNet (ViT-based) + Reinhard colour harmonization
- Grad-CAM model interpretability

## Quick Start — Server for Android App

```bash
cd placement_app
pip install fastapi uvicorn
python server.py --port 8000
# → http://<your-ip>:8000/docs  (Swagger UI)
```

Endpoints: `/api/place`, `/api/place_manual`, `/api/harmonize`, `/api/mask`, `/api/health`

## Training — TopNet (Fixed Transformer)

### Data Preparation

Dataset and pretrained weights are available at:

> **SJTU Cloud Drive**: [
https://pan.sjtu.edu.cn/web/share/7d842afe7e9850ff1d5453dc62a19a7a](
https://pan.sjtu.edu.cn/web/share/7d842afe7e9850ff1d5453dc62a19a7a)(code: njb3)

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

### Ablation Experiments

| Exp | Config | Model Architecture | Loss | Description |
|-----|--------|-------------------|------|-------------|
| A | `stage1_ce.yaml` | 79M, 2ch logits | CrossEntropy(ignore=255) | Sparse CE baseline |
| A2 | `stage1_ce_dilated.yaml` | 79M, 2ch logits | CrossEntropy(ignore=255) | Label dilation (r=3px disks) |
| B | `stage1_focal.yaml` | 79M, 1ch heatmap | Focal (partial, alpha=2, beta=4) | Gaussian peaks, pos+neg supervised only |
| B-113M | `stage1_focal_113M.yaml` | 113M, 1ch heatmap | Focal (partial) | Same as B, scaled-up model |
| C | `stage1_focal_full.yaml` | 79M, 1ch heatmap | Focal (full, pos_weight=50) | All pixels supervised (abandoned) |
| C-113M | `stage1_focal_full_113M.yaml` | 113M, 1ch heatmap | Focal (full) | Same as C, scaled-up (abandoned) |

### Run Training

```bash
cd topnet_fixed
bash sh_scripts/expA_ce.sh              # CE baseline
bash sh_scripts/expB_focal.sh           # Focal partial (79M)
bash sh_scripts/expB_focal_113M.sh      # Focal partial (113M)
```

### Evaluation

```bash
# Full test set (2568 images)
python test_exp.py --device cuda

# Quick check (first 4 batches)
python test_exp.py --quick --device cuda
```

### Key Results

All models evaluated on the full test set (sparse F1/bAcc, 2568 images):

| Model | F1 | bAcc | Params |
|-------|-----|------|--------|
| Buggy TopNet (original) | 0.658 | 0.753 | 113.2M |
| CE baseline (ours) | 0.660 | 0.750 | 79.4M |
| Focal partial (ours) | **0.672** | 0.764 | 79.4M |
| Focal partial 113M (ours) | **0.672** | **0.766** | 113.0M |

Reference (original TopNet paper): F1=0.741, bAcc=0.815

## Model Architecture

### SimOPA (Object Placement Assessment)
- 4-channel ResNet18 (RGB + mask) → GAP → FC(512, 2)
- 11.2M params, scores a single composite image

### TopNet / ObPlaNet (Fixed, CVPR 2023)
- Dual 4-channel ResNet18 encoders (SOPA-pretrained bg, ImageNet-pretrained fg)
- 4-layer Transformer with per-token self-attention (8 heads, 1024-dim)
- UNet decoder with background skip connections
- **Bug fixes** vs original:
  - LayerNorm(1024) on feature dimension (was LayerNorm(8) on spatial columns)
  - `batch_first=True` for correct cross-position attention (was treating batch as sequence)
  - Per-token MLP (was 65536→128→65536 global FC, 67M params)
  - 79.4M params (standard) / 113.0M (scaled-up, mlp_expansion=8)
- Output: `[B, 2, 256, 256]` (binary) or `[B, 1, 256, 256]` (keypoint heatmap)

### PCTNet (Image Harmonization)
- ViT encoder-decoder + Pixel Color Transform
- 4.8M params, ViT-based per-pixel colour transform
- Fallback: Reinhard colour transfer (Lab-space statistics matching)

## Training Design

### Two-stage Strategy
- **Stage 1**: Freeze both encoders (SOPA + ImageNet), train Transformer + Decoder only
  - lr=1e-4, batch=32, optimizer=AdamW
- **Stage 2**: Unfreeze all, full-model fine-tuning
  - lr=1e-5, batch=32, optimizer=AdamW

### Loss Functions
- **CrossEntropy**(ignore_index=255): original paper's approach, only sparse labeled pixels contribute
- **Focal Loss**(alpha=2, beta=4): CenterNet-style, applied on Gaussian heatmaps
  - Partial supervision: only pos Gaussian regions + neg windows
  - Normalized by number of annotation peaks (not Gaussian area)

### Data Variants
- **Sparse**: single-pixel labels (original, ~6 labeled px/image)
- **Gaussian**: 2D Gaussian peaks centered at pos_label, sigma = min(fg_w,fg_h) / 8
- **Dilated**: label points expanded to radius-3 disks (~174 labeled px/image), compatible with CE

## References

- OPA: [Object Placement Assessment Dataset](https://arxiv.org/abs/2107.01889) (arXiv 2021)
- TopNet: [Transformer-based Object Placement Network](https://github.com/bcmi/TopNet-Object-Placement) (CVPR 2023)
- SOPA / FOPA: [BCMI Lab](https://github.com/bcmi/libcom)
- CenterNet: [Objects as Points](https://arxiv.org/abs/1904.07850) (Zhou et al., 2019)
- SAM2: [Segment Anything 2](https://github.com/facebookresearch/sam2) (Meta)
- PCTNet / Reinhard: from [libcom](https://github.com/bcmi/libcom) harmonization toolkit

## Model Interpretability

The Gradio app provides five interpretability views on the top-ranked placement result:

| Tab | Method | What it shows |
|-----|--------|---------------|
| Grad-CAM | Gradient-weighted CAM on SimOPA | Which image regions the model focuses on |
| Saliency | Input gradient magnitude | Which pixels most influence the score |
| Occlusion | Sliding gray-window score drop | Where occlusion causes the largest score decrease |
| Features L2 | ResNet layer2 activations | Mid-level features (textures, edges) |
| Features L4 | ResNet layer4 activations | Deep features (semantics, layout) |

## Server for Android App

```bash
cd placement_app
pip install fastapi uvicorn
python server.py --port 8000
# → http://<your-ip>:8000/docs
```

REST API: `/api/place`, `/api/place_manual`, `/api/harmonize`, `/api/mask`, `/api/health`

## License

MIT
