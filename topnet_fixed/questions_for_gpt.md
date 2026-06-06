I am reproducing and fixing TopNet (CVPR 2023, Transformer-based Object Placement Network).
The original code has two bugs in its Transformer block. I fixed them and am training
from scratch, but results are consistently worse than the buggy version. Please help
me diagnose the root cause(s).

**Task**: Given a background image + foreground object + mask, output a 256×256
heatmap where each pixel = P(placing the foreground center here is reasonable).
Training labels are SPARSE point annotations: pos_label (reasonable center points)
and neg_label (unreasonable). Only ~1-10 pixels per 256×256 image have labels;
99.99% of pixels are unlabeled (ignore=255). The original paper uses
CrossEntropyLoss(ignore_index=255) and reports F1=0.741, bAcc=0.815 (evaluated
ONLY on the sparse labeled pixels).

---

## Buggy vs Fixed Architecture

Both use dual 4ch ResNet18 encoders + 4-layer Transformer + UNet decoder.

**Buggy (113M params, the version that achieves F1=0.741)**:
- LayerNorm(8) applied directly on [B, 1024, 8, 8] — normalizes across spatial
  columns (W=8), NOT the feature dimension
- MHA input: [B, 64, 1024] with batch_first=False → MHA treats B as sequence
  length, 64 as batch size. With B>1 during training, attention leaks info
  ACROSS different images in the batch. With B=1 at inference, attention is
  single-token identity (does nothing)
- MLP: Linear(65536 → 128 → 65536), flattening the entire 1024×8×8 feature map.
  This is effectively a 67M-parameter fully-connected layer across all 65K pixels

**Fixed (79M params, my version)**:
- LayerNorm(1024) on the feature dimension (correct)
- MHA with batch_first=True, [B, 64, 1024] → proper cross-position attention
- Per-token MLP: Linear(1024 → 4096 → 1024), standard Transformer design
- Also created a 113M scaled-up version: MLP hidden 4096→8192

---

## Training Results (all with SOPA-pretrained bg_encoder + ImageNet-pretrained fg_encoder)

### Experiment A: Sparse CrossEntropy (reproducing original)
- Same CE loss, same sparse targets
- Fixed 79M model: F1=0.69, bAcc=0.77 (close to original 0.74/0.82)
- CE val_loss drops then overfits in stage2 (encoders unfrozen)

### Experiment B: Focal Loss + Partial Gaussian Heatmap
- Convert sparse pos_label to 2D Gaussian peaks: sigma = min(fg_w, fg_h) / 8
- Convert sparse neg_label to explicit zeros (5px supervision window)
- valid_mask: 1 only at pos Gaussian regions + neg windows, 0 elsewhere
  (i.e., do NOT punish unlabeled pixels — "we don't know")
- Focal Loss(alpha=2, beta=4) computed ONLY on supervised regions
- Model output: [B, 1, 256, 256] with Sigmoid
- Validation: eval on SAME sparse annotations (threshold heatmap at 0.5, compare
  to sparse pos/neg labels)
- Results with sigma=8.0:
  - 79M: F1=0.49
  - 113M: F1=0.58
  - Both significantly worse than CE baseline F1=0.69

### Experiment C: Focal Loss + Full-supervision
- Same Gaussian heatmap but ALL pixels supervised (pos=Gaussian, rest=0)
- No valid_mask — model forced to predict 0 everywhere except Gaussian regions
- With Focal alpha=2, beta=4: model collapses, predicts all zeros, F1=0.000
- Added pos_weight=50: not yet tested

---

## Key Questions

1. **Why does the buggy Transformer work at all?** With B>1 during training, MHA
   leaks info across images (each spatial position attends across the batch).
   With B=1 inference, attention is identity. Yet the model still achieves
   F1=0.74. Is the 67M MLP (65536→128→65536) doing all the heavy lifting by
   memorizing spatial patterns? Or is the cross-batch information leakage actually
   helpful during training?

2. **Why does sparse CE (F1=0.69) beat Focal + Gaussian (F1=0.49-0.58)?**
   Is it because:
   - Gaussian peaks make the target "fuzzy" while the evaluation metric requires
     hitting EXACT pixel positions?
   - The BCE threshold (0.5) on sigmoid output is poorly calibrated for this task?
   - Focal Loss hyperparameters (alpha=2, beta=4) are wrong for 1:650 positive
     imbalance?
   - The per-token MLP doesn't have enough capacity compared to the buggy
     65536→128→65536?

3. **Is evaluating on sparse point annotations fundamentally flawed?**
   With only 1-10 labeled pixels out of 65536, a model that predicts PERFECTLY
   but is 1 pixel off gets F1=0. Would a different evaluation metric or a
   downstream task evaluation (e.g., SimOPA scoring of recommended placements)
   better reflect real model quality?

4. **How should I design the loss for extremely sparse placement annotations?**
   Currently trying:
   - Sparse CE (ignore=255) — works OK but overfits
   - Partial Focal on Gaussian heatmap — underperforms CE
   - Full Focal on Gaussian heatmap — model collapse
   
   What loss + target design would you recommend for this scenario?

5. **Is it worth scaling the per-token MLP to match the buggy model's 67M
   parameter count?** The buggy MLP does 65536→128→65536, which is essentially
   a learnable lookup table for the entire feature map. A per-token MLP with
   the same parameter budget would need hidden_dim ~ 34000 (34000*1024*2 ≈ 70M).
   Is this even meaningful, or would it just overfit?

6. **Gaussian sigma sensitivity**: with sigma ≈ 17px (sigma_factor=6.0), F1=0.66
   (close to CE). With sigma ≈ 8px (sigma_factor=12.0), F1=0.49. Why would a
   wider, fuzzier target produce BETTER evaluation on sparse point annotations?
   Is this evidence that the metric rewards "blurry" predictions that cover more
   area, rather than precise localization?
