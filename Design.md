# Design.md — OpenImageComp 技术分析

## 0. OpenImageComp: 本项目概况

基于 BCMI 实验室的图像合成研究，我们构建了 **OpenImageComp** — 一个整合物体放置评估、深度协调和交互式应用的工具箱。

### 项目结构

| 子项目 | 说明 |
|--------|------|
| `placement_app/` | **Gradio 交互应用** — 物体放置助手，支持 TopNet 热力图搜索 / 网格枚举 + SimOPA 评分 / 手动点击放置，PCTNet/Reinhard 协调，SAM2/OpenCV 自动分割 |
| `topnet_fixed/` | **TopNet 修复版** — 修正原始 Transformer 的两处 bug（LayerNorm 维度 + MHA batch_first），支持 3 组消融实验（稀疏 CE / 局部 Focal / 全图 Focal），79M 参数 |
| `OPA/` | 原始 OPA 参考代码 |
| `TopNet-Object-Placement-main/` | 原始 TopNet 参考代码 |
| `libcom-main/` | libcom 参考工具箱 |

### 关键改进

1. **TopNet Transformer 修复**: 原始代码 LayerNorm(8) 在空间维度归一化而非特征维度，MHA 把 batch 当成序列长度导致 B=1 推理时注意力退化。修复后使用正确的 LayerNorm(1024) + batch_first=True + per-token MLP。

2. **CenterNet 风格改造**: 从稀疏逐像素分类扩展到高斯热力图 + Focal Loss，支持局部监督（仅 pos+neg 标注区域）和全图监督两种模式。

3. **交互应用**: Gradio Web 前端，点击即可放置、TopNet 热力图搜索、SimOPA 精排、PCTNet 协调、SAM2 自动分割。

---

## 1. 项目与论文

| 项目 | 论文 | 会议 | 机构 |
|------|------|------|------|
| **OPA** | "OPA: Object Placement Assessment Dataset" (arXiv 2107.01889) | arXiv 2021 | BCMI Lab (上海交大/华东师大) |
| **TopNet** | "TopNet: Transformer-based Object Placement Network for Image Compositing" | CVPR 2023 | BCMI Lab |
| **libcom** | "Making Images Real Again: A Comprehensive Survey on Deep Image Composition" (arXiv 2106.14490) | arXiv 2021 | BCMI Lab |

三个项目来自同一个 BCMI 实验室，libcom 是该实验室的图像合成统一工具箱，集成了 OPA 的 SimOPA 模型和 FOPA 模型，同时也包含了图像协调、阴影生成、反射生成等多个子任务。

**论文 PDF 位于 `project/papers/`**:
- `OPA Object Placement Assessment Dataset.pdf`
- `TopNet Transformer-based Object Placement Network for Image Compositing.pdf`

---

## 1. 图像合成任务全景

图像合成（Image Composition）的核心流程是：给定一张背景图像、一个前景物体（含遮罩），将前景放置在背景的合适位置和尺度上，使其看起来自然。这涉及多个子任务：

```
前景物体 + 背景图像
    │
    ├─ 1. 放置评估 (OPA)        → "放在这里合理吗？"      → 0~1 评分
    ├─ 2. 放置预测 (TopNet/FOPA) → "应该放在哪里？"       → 位置 + 尺度的热力图
    ├─ 3. 前景搜索 (FOS Score)   → "这个前景和这个背景搭吗？" → 兼容性评分 (给定 bbox)
    ├─ 4. 图像协调 (Harmonization) → "前景颜色不搭怎么办？" → 颜色调整后的图像
    ├─ 5. 阴影/反射生成           → "缺少阴影怎么办？"     → 带阴影/反射的图像
    └─ 6. 风格协调 (Painterly)   → "艺术风格不匹配？"     → 风格化调整后的图像
```

三个项目的分工：
- **OPA**：做第 1 件事——输入合成图像 + 遮罩，输出合理性评分（0~1）
- **TopNet**：做第 2 件事——输入背景 + 前景 + 遮罩，输出全图每个像素位置的合理性热力图
- **libcom**：做第 1~6 全部事情——是一个完整工具箱，集成了 OPA 的 SimOPA 模型（OPAScoreModel）、FOPA 模型（FOPAHeatMapModel），以及 Harmonization、Shadow、Reflection 等多个模块

---

## 1.1 关键概念辨析: SOPA、FOPA、SimOPA 与 TopNet 的关系

### 1.1.1 SOPA 和 FOPA 的模型代码在哪里？

BCMI 实验室围绕"物体放置"这个主题先后发表了多篇论文，代码逐步演进。三项目中各模型的存在情况：

| 模型 | 全称 | 论文 | 模型代码所在项目 |
|------|------|------|-----------------|
| **SOPA** | Slow Object Placement Assessment | CVPR 2022 | ⚠️ 无独立项目，但代码和权重被广泛复用 |
| **SimOPA** | ≡ SOPA 的简化版（Simple OPA） | OPA 论文 (arXiv 2021) | ✅ `OPA/`（训练+推理）、`libcom/opa_score/`（仅推理） |
| **SimOPA-ext** | SimOPA + Faster R-CNN 空间关系推理 | OPA 论文 | ✅ `OPA/`（`simopa_ext_net.py` + `eval_opascore/simopa_ext.py`） |
| **FOPA** | Fast Object Placement Assessment | ECCV 2022 | ✅ `libcom/fopa_heat_map/`（仅推理，无训练代码） |
| **TopNet** | Transformer-based Object Placement Network | CVPR 2023 | ✅ `TopNet-Object-Placement-main/`（训练+推理） |

**关键结论**:
- **SOPA 和 SimOPA 是同一个模型**。"SimOPA" 是 OPA 论文中基线模型的名字（Simple OPA），其实就是 SOPA 的简化实现。SOPA 作为独立论文 (CVPR 2022) 先发表，而 SimOPA 是 OPA 数据集论文 (arXiv 2021) 中对同一架构的称呼。两者的核心架构相同，SimOPA 的代码也在其他项目中被当作 "SOPA" 来引用。
- **SOPA 没有独立的单独项目目录**，但它的模型架构和预训练权重贯穿了所有三个项目：
  - `OPA/object_place_net.py` 中的 `ObjectPlaceNet` 就是 SimOPA/SOPA 的训练实现
  - `libcom/opa_score/` 中的 `OPAScoreModel` 是 SimOPA/SOPA 的推理封装
  - TopNet 的 `backbone/ResNet.py` 加载 `SOPA.pth.tar` 作为背景编码器预训练
  - libcom 的 `fopa_heat_map/` 加载 `SOPA.pth` 作为 FOPA 编码器预训练
- **FOPA 的训练代码不在此三项目中**（FOPA 有独立的 GitHub 仓库: `https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment`），但 libcom 中包含了 FOPA 的推理封装 (`FOPAHeatMapModel`)。
- **TopNet 项目内有一个 FOPA 风格的简化模型**：`network/ObPlaNet_simple.py` 使用动态卷积（`simpleDFN`）融合特征，这与 FOPA 的核心思路一致。这个简化版是 TopNet 的消融实验变体，不是完整的 FOPA 实现。

### 1.1.2 SOPA → FOPA → TopNet 的演进关系

三者的论文都来自 BCMI 同一实验室，在方法上逐步演进：

```
SOPA (CVPR 2022, Slow Object Placement Assessment)
  │  核心贡献:
  │  - 4 通道 ResNet backbone (RGB + mask)
  │  - 输入: 合成图 → 输出: 整图单次评分
  │  - 问题: 只能评估已合成好的图像，无法预测最佳放置位置
  │
  ├──→ FOPA (ECCV 2022, Fast Object Placement Assessment)
  │      改进:
  │      - 输入改为: 背景 + 前景 + mask（不需要先合成）
  │      - 输出改为: 全图像素级合理性热力图（可以找最佳位置）
  │      - 架构: 双流 ResNet 编码器 + 动态卷积融合（simpleDFN）
  │      - 速度优势: 一次前向即可得到所有位置的评分
  │      - "SOPA" 是慢速版（需要逐个位置合成再评估）
  │      - "FOPA" 是快速版（一次推理得到所有位置评分）
  │
  └──→ TopNet (CVPR 2023, Transformer-based Object Placement Network)
          改进:
          - 在 FOPA 的双流编码器架构基础上
          - 用 4 层 Multi-Head Self-Attention Transformer 替代动态卷积
          - Transformer 更好地建模前景-背景的全局空间关系
          - 但速度更慢 (26M 参数 vs FOPA 更轻量)
          - 另外提供了 ViT 编码器变体 (备用方案, 未启用)
```

**技术演进要点**:

| 维度 | SOPA (SimOPA) | FOPA | TopNet |
|------|:---:|:---:|:---:|
| 输入 | 合成图 + 遮罩 | 背景 + 前景 + 遮罩 | 背景 + 前景 + 遮罩 |
| 输出 | 单个评分值 | 全图像素热力图 | 全图像素热力图 |
| 编码器 | 单个 4ch ResNet | 双流 4ch ResNet | 双流 4ch ResNet |
| 特征融合 | 无融合 (仅 GAP) | 动态卷积 (simpleDFN) | Transformer (4层 8头) |
| 速度 | 慢 (需枚举位置) | 快 (单次推理) | 中等 |
| 参数量 | ~11M | ~22M | ~26M |
| 对 SOPA 的依赖 | — (即 SOPA 本身) | 使用 SOPA 权重初始化编码器 | 使用 SOPA 权重初始化 bg_encoder |

### 1.1.3 FOS Score（前景搜索评分）的模型

**FOS Score = Foreground Object Search Score**，来自 **DiscoFOS** 论文 (BCMI 实验室)。

**核心问题**: FOS Score 的输入包含 bbox，那它跟 OPA 有什么区别？bbox 在这个任务中的作用是什么？

**先说结论**: bbox 提供的是**评分所需的空间上下文（尺度+位置）**——模型需要知道前景放多大、放哪里，才能合成出待评估的图像区域。FOS Score 的**主要使用场景是前景物体检索**（给定一个背景和 bbox，从候选池中找出最匹配的前景），而不是放置位置优化。典型用法是**固定 bbox，变前景**来比较不同物体与背景的兼容性——此时位置/尺度一致，分数的差异就反映了语义+风格兼容性。

#### 完整推理流程（基于 `fos_score_prediction.py` 源码）

```
输入:
  background_image  : 背景图像
  foreground_image  : 前景物体图像
  bounding_box      : [x1, y1, x2, y2]
  foreground_mask   : (可选) 前景遮罩

Step 1: 合成裁剪区域 (inputs_preprocess)
  ┌──────────────────────────────────────────────────┐
  │  ① 前景 resize 到 bbox 尺寸:                      │
  │     fg = cv2.resize(fg, (x2-x1, y2-y1))          │
  │     → bbox 的宽高决定了前景在场景中的视觉尺度       │
  │                                                  │
  │  ② 若有 mask: 遮罩外的像素填充为 128 (灰色)         │
  │     → 排除前景图背景的干扰                         │
  │                                                  │
  │  ③ 将前景贴到背景的 (x1,y1) 处:                    │
  │     bg[y1:y2, x1:x2] = fg_resized               │
  │     → bbox 的左上角决定了前景空间位置               │
  │                                                  │
  │  ④ 带上下文外扩的裁剪 (get_crop_bbox):             │
  │     add_w = (x2-x1) × (√2-1) / 2  ≈ 宽×0.207     │
  │     add_h = (y2-y1) × (√2-1) / 2  ≈ 高×0.207     │
  │     裁剪区域 = [x1-add_w, y1-add_h,               │
  │                 x2+add_w, y2+add_h]               │
  │     → 每个方向扩展约 20.7%，让模型"看到"周围上下文  │
  │     → clamp 到背景边界内                          │
  │                                                  │
  │  ⑤ Resize → 224×224                              │
  └───────────────────────┬──────────────────────────┘
                          ↓
Step 2: 归一化 (prepare_input_disc)
  ┌──────────────────────────────────────────────────┐
  │  ToTensor + Normalize(mean=0.5, std=0.5)          │
  │  → [1, 3, 224, 224], 值域 [-1, 1]               │
  └───────────────────────┬──────────────────────────┘
                          ↓
Step 3: VGG19 判别器评分 (SingleScaleD)
  ┌──────────────────────────────────────────────────┐
  │  torchvision.models.vgg19 (参数冻结)               │
  │  分 3 段提取特征:                                  │
  │    backbone[:19]  → 浅层纹理特征                   │
  │    backbone[19:28] → 中层部件特征                   │
  │    backbone[28:]  → 深层语义特征 (block5)           │
  │       │                                          │
  │       └→ GAP(block5_feat) → 512-d                │
  │          Linear(512 → 1) → Sigmoid               │
  │          → score ∈ [0, 1]                        │
  └───────────────────────┬──────────────────────────┘
                          ↓
输出: fos_score (float) — 越高越兼容
```

#### bbox 的三重作用

| bbox 参数 | 决定了什么 | 物理意义 |
|-----------|-----------|----------|
| `w = x2-x1`, `h = y2-y1` | 前景尺度 | 前景缩放到多大——大 bbox = 大物体，小 bbox = 小物体 |
| `x1, y1` | 合成位置 | 前景放在背景的哪个坐标 |
| bbox 区域 × √2 | 评估窗口 | 模型看的范围比 bbox 大 41%，捕获前景周围的环境上下文 |

#### 与 OPA/FOPA 的对比

| | OPA (SimOPA) | FOS Score (DiscoFOS) |
|------|-------------|---------------------|
| 输入 | 已合成好的整张图 + mask | bg + fg + bbox (+ 可选 mask) |
| 合成谁做 | 外部预先合成 | **内部合成** (resize + paste) |
| 骨干网络 | 4ch ResNet18 + FC(512,2) | 冻结 VGG19 + FC(512,1)→Sigmoid |
| 训练方式 | 二分类 CE Loss | 判别器 + Triplet Loss(margin=0.1) + 知识蒸馏 |
| 评估什么 | "这个合成图自然吗" | "这个前景跟这个背景/尺度搭吗" |
| 主要用途 | 评估已知放置的合理性 | **前景物体检索/排序** (固定 bbox, 变前景比较) |
| 评估指标 | F1, bAcc | Recall@K, mAP, Precision@K (检索指标) |

#### 训练时使用的 StudentModel（知识蒸馏变体）

推理只用 `SingleScaleD`，但训练时还有一个**双流蒸馏网络**：

```
StudentModel
  ┌──────────────┐   ┌──────────────┐
  │ bg_encoder    │   │ fg_encoder    │
  │ VGG19 分3段   │   │ VGG19 分3段   │  ← 两者共享权重
  └──────┬───────┘   └──────┬───────┘
         │                  │
         └────────┬─────────┘
                  ↓
     Distillation Module (6 种融合方式可选):
     roiconcat / roicropresize / globalconcat /
     roivectorconcat / gapconcat / onlyclassify
                  ↓
     AuxClassifier: GAP + Conv(1,1) + Sigmoid → score
                  ↓
     训练损失 = BCE + 蒸馏损失(L1/L2/SmoothL1) + Triplet Loss(margin=0.1)
```

**权重**: `FOS_D.pth` (~550 MB)  
**训练集**: FOSD (Foreground Object Search Dataset)

---

## 2. OPA（Object Placement Assessment）

### 2.1 任务背景

OPA 解决的是**合成图像合理性评估**问题。给定一张已经合成好的图像（composite image）和前景物体的遮罩，判断这个放置是否"合理"——即前景的大小、位置、遮挡关系、语义匹配是否看起来自然。

这是一个**二分类任务**（合理=1，不合理=0），输出为一个 [0, 1] 的置信度评分。

### 2.2 深度学习模型

#### 2.2.1 SimOPA（基线模型）

```
输入: [B, 4, 256, 256]  ← RGB图像 (3通道) ⊕ 前景遮罩 (1通道)
       │
       ▼
┌─────────────────────────────────┐
│  4通道 ResNet18 (Backbone)       │
│  - conv1: Conv2d(4→64)           │
│  - layer1~layer4 标准 ResNet     │
│  - 去掉最后的 avgpool 和 fc      │
│  输出: [B, 512, 8, 8]           │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  AdaptiveAvgPool2d(1)           │
│  输出: [B, 512]                 │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Linear(512, 2)                 │
│  输出: logits [B, 2]           │
└──────────────┬──────────────────┘
               ▼
         softmax → score ∈ [0, 1]
```

**关键设计**:
- **4 通道输入**: 第一个卷积层 `Conv2d(4, 64)` 而非标准 `Conv2d(3, 64)`
- **遮罩通道初始化技巧**: 第 4 通道的权重用 RGB→灰度公式初始化: `W₄ = 0.299·Wᵣ + 0.587·W_g + 0.114·W_b`，而非随机初始化
- **参数量**: ResNet18 骨干约 11M 参数
- **配置文件**: `config.py`（`Config` 类，全局单例 `opt`）
- **预训练**: ResNet18 ImageNet 预训练 + 微调

#### 2.2.2 SimOPA-ext（扩展模型，引入空间关系推理）

在 SimOPA 基础上增加了**前景-背景空间关系建模**能力：

```
输入:
  - composite: [B, 4, 256, 256]          (RGB+mak 拼接)
  - target_box: [B, 4]                   (前景边界框)
  - refer_feat: [B, 5, 2048]             (前5个参考物体的 Faster R-CNN 特征)
  - target_feat: [B, 2048]               (前景物体的 Faster R-CNN 特征)
  - binary_mask pair: [B, 2, 64, 64]     (前景&参考物体的几何遮罩对)
       │
       ▼
┌─────────────────────────────────────────────┐
│  4通道 ResNet18 (Backbone, 与 SimOPA 相同)   │
│  输出: [B, 512, 8, 8]                       │
└──────────┬──────────────────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
  Global        Regional Feature
  Feature       Extraction
  [B,512]       │
                ├─ RoI Align 从特征图提取参考物体区域 (3×3)
                ├─ 线性投影 → 512-d
                ├─ 与目标物体特征拼接
                ├─ fc → 1024-d
                │
                └─ Geometric Feature (几何特征提取器):
                   Conv(2→64, s2)→ReLU→MaxPool→Conv(64→256, s2)→ReLU→MaxPool
                   → Flatten → Linear(4096→256) → [B, N, 256]
                │
                ▼
           ┌─────────────────────────────┐
           │  Multi-Head Self-Attention   │
           │  - 16 heads × 64 dim/head    │
           │  - 在 N 个参考物体之间建模关系 │
           │  - Attention pooling 聚合     │
           └──────────┬──────────────────┘
                      ▼
                concat([global_feat, agg_region_feat])
                → [B, 1536]
                      ▼
                MLP: 1536 → 1536 → 512 → 2
                (含 Dropout(0.1) 正则化)
                      ▼
                logits [B, 2] → softmax → score
```

**核心创新**:
1. **Faster R-CNN 特征提取**: 使用在 Visual Genome (1600 类) 上预训练的 Faster R-CNN (ResNet-101) 检测背景中的参考物体并提取 2048-d 特征
2. **几何特征编码**: 用一个小 CNN 将目标和参考物体的边界框遮罩编码为 256-d 几何特征
3. **多头自注意力**: 在多个参考物体之间建模关系（16 头，64 维/头）
4. **可解释性**: 注意力权重可输出每个参考物体对决策的贡献

### 2.3 非深度学习 CV 算法

| 算法 | 所在文件 | 功能 |
|------|----------|------|
| Alpha 混合 | `data_processing/generate_composite.py` | `composite = foreground × mask + background × (1 - mask)` |
| 遮罩→边界框 | `eval_opascore/simopa_ext.py` → `mask2bbox()` | `np.where(mask >= 127)` 找到非零像素 → 计算 min/max 得到 [x1,y1,x2,y2] |
| 图像读写与缩放 | 所有推理脚本 | PIL.Image.open / resize / ToTensor |
| 推理时图像预处理 | simopa.py / simopa_ext.py | Resize 256×256, 归一化, 通道拼接 |

### 2.4 数据集与预训练权重

#### OPA 数据集
| 属性 | 值 |
|------|------|
| 总样本数 | 73,470 |
| 训练集 | 62,074 (正样本 21,376 + 负样本 40,698) |
| 测试集 | 11,396 (正样本 3,588 + 负样本 7,808) |
| 前景物体 (train/test) | 2,701 / 1,436 (无重叠) |
| 背景图像 (train/test) | 1,236 / 153 (无重叠) |
| 数据格式 | CSV + 图像文件 (composite, mask) |
| 数据来源 | 基于 COCO 数据集生成 |

**OPA-ext** (扩展数据集, 80,263 样本): 标注质量较差，与测试集有前景/背景泄漏——**不可用于增强训练**。

#### 数据集下载

| 内容 | 百度云 (提取码) | Dropbox |
|------|-----------------|---------|
| OPA 数据集 | 百度云 `a982` | Dropbox |
| OPA-ext 数据集 | 百度云 `fogy` | Dropbox |

#### 预训练权重下载

| 权重 | 大小 | 百度云 (提取码) | Dropbox |
|------|------|-----------------|---------|
| best-acc.pth (SimOPA) | ~44.8 MB | 百度云 `up1c` | Dropbox |
| 基础 ResNet18 | ~44.8 MB | 百度云 `msqg` | Dropbox |
| simopa.pth + simopa_ext.pth | 2×~44.8 MB | 百度云 `bcmi` | Dropbox |
| faster_rcnn_res101_vg.pth | ~850 MB | N/A | Google Drive / Dropbox |

**本地已存在**: `eval_opascore/checkpoints/simopa.pth`, `simopa_ext.pth`（各 44.8 MB）

#### PyTorch 标准预训练（代码内自动下载）
```
https://download.pytorch.org/models/resnet18-5c106cde.pth
https://download.pytorch.org/models/resnet34-333f7ec4.pth
https://download.pytorch.org/models/resnet50-19c8e357.pth
https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
https://download.pytorch.org/models/resnet152-b121ed2d.pth
```

---

## 3. TopNet（Transformer-based Object Placement Network）

### 3.1 任务背景

TopNet 解决的是**物体放置预测**问题——给定背景图像和前景物体（含遮罩），预测前景应该放在背景的**哪个位置和什么尺度**才算合理。

与 OPA 的区别：
- OPA: 输入=已合成的图像，输出=这个放置是否合理的评分（**评估已知放置**）
- TopNet: 输入=背景+前景+遮罩，输出=全图像素级别的合理性热力图（**预测最佳放置位置**）

这是一个**逐像素二分类**任务，生成一张与输入同大小（256×256）的评分图，每个像素值表示在该位置放置前景是否合理。

### 3.2 深度学习模型

#### 3.2.1 ObPlaNet_resnet18（主模型）

```
输入:
  - bg:    [B, 3, 256, 256]   背景图像
  - fg:    [B, 3, 256, 256]   前景物体图像
  - mask:  [B, 1, 256, 256]   前景遮罩
       │
       ├──────────────────┐
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│ bg_encoder    │   │ fg_encoder    │
│ (ResNet18,    │   │ (ResNet18,    │
│  4ch, 输入=    │   │  4ch, 输入=    │
│  RGB⊕空遮罩)  │   │  RGB⊕前景遮罩) │
│               │   │               │
│ SOPA 预训练    │   │ ImageNet 预训练│
│ 参数冻结? 否   │   │ 第4通道灰度初始化│
│               │   │               │
│ → 5层特征:    │   │ → 6层特征:    │
│   div2:  [64] │   │   div1:  [64] │
│   div4:  [64] │   │   div2:  [64] │
│   div8: [128] │   │   div4:  [64] │
│   div16:[256] │   │   div8: [128] │
│   div32:[512] │   │   div16:[256] │
│               │   │   div32:[512] │
└──────┬───────┘   └──────┬───────┘
       │                  │
       │  bg_div32 [B,512,8,8]
       │  fg_div32 [B,512,8,8]
       │                  │
       └────┬─────────────┘
            ▼
    cat → [B, 1024, 8, 8]
            │
            ▼
┌───────────────────────────────┐
│  Transformer Module (4层)     │
│  - MultiheadAttention(1024,   │
│       num_heads=8)            │
│  - LayerNorm                  │
│  - MLP: 65536 → 128 → 65536   │
│  输出: [B, 1024, 8, 8]       │
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────────────┐
│  UNet-style Decoder (含跳连)           │
│                                       │
│  upconv32: 1024→512  [8×8]            │
│  upconv16: 512→256   + bg_enc16 [16×16]│
│  upconv8:  256→128   + bg_enc8  [32×32]│
│  upconv4:  128→64    + bg_enc4  [64×64]│
│  upconv2:  64→64     + bg_enc2 [128×128]│
│  upconv1:  64→64     [256×256]         │
│                                       │
│  每个 upconv = Conv2d→BN→ReLU + 上采样  │
└──────────────┬────────────────────────┘
               ▼
    ConvTranspose2d(64 → 512, 3×3)
    Conv2d(512 → 2, 1×1)
               ▼
    输出: [B, 2, 256, 256]
    (logits for each pixel: unreasonable / reasonable)
```

**训练细节**:
- **损失函数**: CrossEntropyLoss(ignore_index=255) — 忽略未标注区域的像素
- **优化器**: Adam, lr=1e-5, weight_decay=1e-4
- **学习率调度**: 每 2 轮乘以 0.5 (`all_decay`)
- **训练轮次**: 25 epochs, batch_size=8
- **输入尺寸**: 256×256
- **参数量**: 约 26M (双 ResNet18 编码器 ~22.4M + Transformer ~1.6M + 解码器 ~2M)

#### 3.2.2 network_vit.py（ViT 变体，备用方案）

用 Vision Transformer (ViT-Base/16, patch_size=16, 768-d embedding) 代替 ResNet18 作为编码器：
- 分别编码背景和前景，各取前 6 层 → [B, 196, 768]
- 拼接 → [B, 392, 768] → Linear(768→256) → reshape → [B, 392, 16, 16]
- Transformer: embed=392, heads=8, layers=4
- 最终分类器: Conv2d(512→32, 1×1)

**注意**: 该 ViT 变体在 `network/__init__.py` 中被注释掉（未启用）。

#### 3.2.3 ObPlaNet_simple（简化版/FOPA风格）

使用**动态卷积融合**（`simpleDFN`）代替 Transformer，是 FOPA 架构的变体。具体实现见 `network/ObPlaNet_simple.py`。

### 3.3 非深度学习 CV 算法

TopNet 的训练和推理流程中，除标准图像缩放（双线性插值）、归一化（ImageNet 均值/标准差）外，无其他传统 CV 算法。数据合成使用标准的 Alpha 混合 (`fg × mask + bg × (1 - mask)`)。

`utils/metric.py` 中包含显著性检测指标（F-measure, MAE, S-measure, E-measure, Weighted F-measure），使用 SciPy 的高斯滤波和距离变换——这些是用于评估的非学习算法，不是模型本身的一部分。

### 3.4 数据集与预训练权重

#### TopNet 数据集
- 基于 SOPA/OPA 协议构建
- 包含背景、前景、遮罩三部分图像
- 标注: JSON 格式，包含 `pos_label`（合理位置列表）、`neg_label`（不合理位置列表）、`scale`、`newWidth/Height`
- 数据目录结构: `data/data/bg/`, `fg/`, `mask/` + `train_pair_new.json` / `test_pair_new.json`

#### 数据集下载

| 内容 | 百度云 (提取码) | Dropbox |
|------|-----------------|---------|
| 数据集 (图像+标注) | 百度云 `4zf9` | Dropbox |
| SOPA 编码器权重 | 百度云 `1x3n` | Dropbox |
| TopNet 预训练权重 | 百度云 `jx6u` | Dropbox |

#### 预训练依赖
- **SOPA.pth.tar**: 背景编码器的预训练权重（4 通道 ResNet18），从 SOPA 模型迁移
- **best_weight.pth**: TopNet 完整训练好的权重
- **PyTorch ResNet 标准预训练**: 前景编码器使用 ImageNet 预训练的 ResNet18

---

## 4. libcom（Library of Image Composition）

### 4.1 任务背景

libcom 是 BCMI 实验室开发的**图像合成统一工具箱**（v0.1.7, Apache 2.0），集成了 10+ 个图像合成子任务的深度学习模型。它不是一个单独的算法，而是一个**模型集合**，通过统一的 API 提供图像合成全流程支持。

**在线演示**: http://libcom.ustcnewly.com/

**安装**: `pip install libcom` (PyPI)

### 4.2 各模块的深度学习模型

libcom 的架构设计高度模块化——每个子任务是一个独立的 `*Model` 类，内部封装完整的模型加载、预处理、推理流程。所有预训练权重从 HuggingFace (`BCMIZB/Libcom_pretrained_models`) 或 ModelScope (`yujieouo/Libcom_pretrained_models`) 自动下载。

#### 4.2.1 OPAScoreModel — 物体放置合理性评估 (≡ SimOPA)

```
输入: composite_image (RGB) + composite_mask (灰度)
       ↓ 拼接为 [B, 4, 256, 256]
ObjectPlaceNet (4ch ResNet18 → GAP → Linear(512, 2))
       ↓ softmax
输出: opa_score ∈ [0, 1]
```
直接复用 OPA 项目的 SimOPA 架构。权重: `SimOPA.pth`

#### 4.2.2 FOPAHeatMapModel — 快速物体放置热力图

```
输入: background + foreground + mask
       ↓ 生成多尺度前景 (16 个尺度, 1/18 ~ 16/18)
每个尺度:
  ObPlaNet_resnet18 (双流编码器 + 动态卷积融合)
       ↓
  heatmap: [2, 256, 256] → 逐像素合理性评分
       ↓ 取 top-50 位置
输出: box_list (最佳放置的边界框) + heatmap_list
```
这是 FOPA (ECCV 2022) 的推理实现。权重: `FOPA.pth`, `SOPA.pth`

**双流编码器架构**:
- 背景编码器 (4ch ResNet18, SOPA 预训练, 冻结参数)
- 前景编码器 (4ch ResNet18, ImageNet + 灰度初始化)
- 特征融合: `simpleDFN` 动态卷积 (从前景特征生成卷积核, 应用到背景特征)
- 解码器: UNet 风格上采样 + 跳连

#### 4.2.3 HarmonyScoreModel — 图像和谐度评分

```
输入: composite_image + composite_mask
       ↓ resize 256×256
StyleEncoder (5 层 Partial Convolution)
  - 分别提取前景区域和背景区域的 16-d 风格向量
  - PartialConv 只对有效区域 (mask=1 或 mask=0) 做卷积
       ↓
harmony_score = exp(-0.04212 × ||bg_style - fg_style||₂)
输出: harmony_score ∈ [0, 1]
```
权重: `BargainNet.pth`。模型来自 BargainNet 项目。

**StyleEncoder 架构**: 5 层 PartialConv2d (stride=2, no padding), 通道: 3→64→128→256→512→512, 最终 AdaptiveAvgPool2d + Conv2d → 16-d

#### 4.2.4 ImageHarmonizationModel — 图像协调（调整前景颜色）

两种模式:

**PCTNet (Pixel-Color-Transform)**:
```
composite + mask (256×256)
       ↓
ViT Encoder (patch embedding + 9 层 Transformer, 2 头, dim=256)
       ↓
Decoder → per-pixel color transform parameters (12-d)
       ↓
Linear PCT: 对每个像素做 3×3 矩阵变换 + 3-d bias
       ↓ 双三次插值到全分辨率
输出: harmonized_image (前景区域颜色调整, 背景不变)
```
权重: `PCTNet.pth`, `IdentityLUT33.txt`

**LBM (Latent Background Matching, 扩散模型)**:
```
composite + mask (1024×1024)
       ↓ VAE Encoder → latent
Diffusion Process (FlowMatchEulerDiscrete, 4 steps)
  - 以 composite image 为 condition
  - 学习匹配背景的 latent 分布
       ↓ VAE Decoder
输出: harmonized_image
```
权重: `lbm_ckpt/` 文件夹

#### 4.2.5 FOSScoreModel — 前景物体搜索评分 (DiscoFOS)

**用途**: 评估前景物体与背景的兼容性，主要用于前景物体检索（固定 bbox，对不同候选前景打分排序）。

**详细分析见 [1.1.3 节](#113-fos-score前景搜索评分的模型)**。以下补充模型架构信息：

**推理模型: `SingleScaleD`** (基于 VGG19 判别器)
- VGG19 骨干分 3 段提取特征 (block1~3 纹理, block4 部件, block5 语义)
- block5 特征 → GAP → 512-d → Linear(512, 1) → Sigmoid → [0, 1]
- VGG 参数冻结，仅 FC 层可训练
- 输入: 224×224 裁剪合成区域 (bbox 外扩 √2 倍后 resize)

**训练时蒸馏变体: `StudentModel`** (双流 VGG19)
- 背景编码器 + 前景编码器 (各为 VGG19 分 3 段)
- 蒸馏融合模块 (6 种: `roiconcat`, `roicropresize`, `globalconcat`, `roivectorconcat`, `gapconcat`, `onlyclassify`)
- 训练损失: BCE + 蒸馏 (L1/L2/SmoothL1) + Triplet Loss (margin=0.1)

**权重**: `FOS_D.pth` (~550 MB)
**训练数据集**: FOSD (Foreground Object Search Dataset)
**评估指标**: Recall@K, mAP, Precision@K (检索指标)

#### 4.2.6 InharmoniousLocalizationModel — 不和谐区域定位

```
输入: composite_image (256×256, ImageNet 归一化)
       ↓
MadisNet (DIRLNet + HDRPointwiseNN 双网络):
  - HDRPointwiseNN: 基于双边网格的颜色重调
    (Coeffs: 3→24→48→96 → ResBlock×3 → Self-Attention → 12×8×16×16 系数网格)
    (GuideNN: 3→16→1 → tanh 引导图)
    (Slice + ApplyCoeffs: 仿射颜色变换)
  - DIRLNet: 不和谐区域检测
    Encoder: ResNet-34/50 → 5 层特征 (224,112,56,28,14)
    Transition: 双向特征整合
    Mask-guided Dual Attention: 通道注意力(ECA) + 空间注意力(SpatialGate)
    Decoder: Global-context Guided Decoder → Sigmoid
       ↓
输出: inharmonious_mask (0-255 uint8)
```
权重: `Inharmonious_G.pth`, `IHDRNet.pth`

#### 4.2.7 其他模块（简要）

| 模块 | 模型 | 输入 | 输出 | 方法 |
|------|------|------|------|------|
| **PainterlyHarmonization** | PHDNet / PHDiffusion | composite + mask | 风格化协调图像 | 专用网络或扩散模型+Adapter |
| **ShadowGeneration** | GPSDiffusion | shadowfree_img + mask | 带阴影的图像 | ControlNet + DDPM |
| **ReflectionGeneration** | RGDiffusion | composite + mask | 带反射的图像 | ControlNet + Diffusion |
| **KontextBlendingHarmonization** | FLUX.1-Kontext-dev + LoRA | bg + fg + bbox + prompt | 融合/协调图像 | 大规模扩散模型 + LoRA fine-tune |
| **OSInsertModel** | ObjectStitch + SAM + InsertAnything | bg + fg + bbox | 生成式插入图像 | 分割+扩散填充 |

### 4.3 非深度学习 CV 算法

libcom 中有两个重要的传统 CV 算法:

#### 4.3.1 Reinhard 颜色迁移 (color_transfer/reinhard.py)

基于 Reinhard 等人 2001 年经典论文的颜色迁移:

```
1. 将图像从 BGR 转换到 CIE Lab* 颜色空间
2. 分别计算背景区域 (mask=0) 和前景区域 (mask=255) 的 Lab* 均值 (μ) 和标准差 (σ)
3. 对前景的每个 Lab* 通道做线性映射:
     ratio = σ_bg / σ_fg
     offset = μ_bg - μ_fg × (σ_bg / σ_fg)
     transformed = original × ratio + offset
4. 转换回 BGR
5. 仅替换前景区域: output = transformed × mask + original × (1 - mask)
```
特点: 无需训练，纯统计方法。计算量小，但对复杂光照变化效果有限。

**论文**: Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001

#### 4.3.2 Naive Composition (naive_composition/generate_composite_image.py)

| 模式 | 算法 | 说明 |
|------|------|------|
| `none` | 直接复制粘贴 | 前景缩放到 bbox 尺寸，直接覆盖背景 |
| `gaussian` | 高斯模糊边缘混合 | 对遮罩做高斯模糊，用模糊后的 alpha 做混合 |
| `poisson` | 泊松融合 | OpenCV `seamlessClone` + `NORMAL_CLONE`，梯度域编辑 |

#### 4.3.3 推理辅助算法

- `mask2bbox`: 从遮罩计算边界框（`np.where(mask >= 127)`）
- `generate_binary_mask`: 生成 64×64 的几何遮罩图
- Central Difference Convolution (`Conv2d_cd`): 增强边缘/梯度的卷积变体，用于不和谐区域检测

### 4.4 数据集与预训练权重

#### libcom 的数据集依赖

libcom 本身**不包含训练代码**——它是纯推理工具箱。但内部模型在以下数据集上训练:

| 模型 | 训练数据集 |
|------|-----------|
| OPAScoreModel | OPA 数据集 (同 OPA 项目) |
| FOPAHeatMapModel | OPA + COCO 数据生成 |
| HarmonyScoreModel | iHarmony4 (IHD) |
| ImageHarmonizationModel | iHarmony4 (IHD) |
| InharmoniousLocalizationModel | iHarmony4 (IHD) |
| FOSScoreModel | FOSD (Foreground Object Search Dataset) |
| ShadowGenerationModel | DESOBA (Shadow) |
| ReflectionGenerationModel | DEROBA (Reflection) |

#### 预训练权重自动下载系统

libcom 的预训练权重通过 `libcom/utils/model_download.py` 统一管理:

1. **HuggingFace Hub**（主源）: `BCMIZB/Libcom_pretrained_models`
2. **ModelScope**（备源）: `yujieouo/Libcom_pretrained_models`

首次使用模型时自动下载，无需手动操作。可通过环境变量 `LIBCOM_MODEL_DIR` 修改存储位置。

#### 完整权重清单

| 权重文件 | 对应模型 | 用途 |
|----------|----------|------|
| `SimOPA.pth` | OPAScoreModel | 放置合理性评分 |
| `FOPA.pth` | FOPAHeatMapModel | 放置热力图生成 |
| `SOPA.pth` | FOPAHeatMapModel (编码器) | 背景编码器预训练 |
| `BargainNet.pth` | HarmonyScoreModel | 和谐度评分 |
| `PCTNet.pth` | ImageHarmonizationModel | 像素颜色变换协调 |
| `IdentityLUT33.txt` | LUT 模块 | 33³ 恒等 LUT |
| `lbm_ckpt/` (文件夹) | ImageHarmonizationModel | 扩散模型 (LBM) |
| `FOS_D.pth` | FOSScoreModel | 前景搜索评分 (~550MB) |
| `Inharmonious_G.pth` | InharmoniousLocalizationModel | 不和谐区域检测 |
| `IHDRNet.pth` | InharmoniousLocalizationModel | HDR 双侧网格 |
| `flux_kontext_blending.safetensors` | KontextBlendingHarmonizationModel | FLUX 融合 LoRA |
| `flux_kontext_harmonization.safetensors` | KontextBlendingHarmonizationModel | FLUX 协调 LoRA |
| `Shadow_cldm.ckpt` | ShadowGenerationModel | 阴影 ControlNet |
| `Shadow_ppp.ckpt` | ShadowGenerationModel | 后处理模块 |
| `Shadow_reg.pth` | ShadowGenerationModel | 回归模块 |
| `Reflection_cldm.ckpt` | ReflectionGenerationModel | 反射 ControlNet |
| `Reflection_ppp.ckpt` | ReflectionGenerationModel | 后处理模块 |
| `Reflection_reg.pth` | ReflectionGenerationModel | 回归模块 |

---

## 5. 三项目关系总结

```
                    ┌─────────────┐
                    │   libcom    │ ← 统一工具箱，集成了所有模型
                    └──────┬──────┘
          ┌────────────────┼────────────────────┐
          ▼                ▼                     ▼
   ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
   │ OPAScoreModel│  │FOPAHeatMapModel│ │HarmonyScoreModel │
   │ (≡ SimOPA)   │  │ (≡ FOPA)      │ │ (≡ BargainNet)    │
   └──────┬──────┘  └──────────────┘  └──────────────────┘
          │
          ▼
   ┌──────────────┐
   │   OPA 项目    │ ← SimOPA 和 SimOPA-ext 的独立训练代码
   └──────────────┘

   ┌──────────────┐
   │ TopNet 项目    │ ← 独立的物体放置预测模型，CVPR 2023
   └──────────────┘
   （TopNet 的评估指标中对比了 FOPA 和 SOPA）

   依赖关系:
   - TopNet 的背景编码器加载 SOPA 预训练权重 (SOPA.pth.tar)
   - libcom 的 FOPAHeatMapModel 加载 SOPA 权重作为编码器
   - libcom 的 OPAScoreModel 直接复用 SimOPA 权重
   - libcom 的 HarmonyScoreModel 来自 BargainNet 项目
```

三个项目都来自 BCMI 实验室，共享以下基础设施:
- **4 通道 ResNet backbone**: OPA → SOPA → TopNet → FOPA 都在使用，遮罩通道初始化公式一致 (0.299R+0.587G+0.114B)
- **OPA 数据集**: OPA 和 TopNet 使用同一套数据标注协议
- **统一在线演示**: http://libcom.ustcnewly.com/

**核心差异**:
| 维度 | OPA | TopNet | libcom |
|------|-----|--------|--------|
| 定位 | 单任务研究项目 | 单任务研究项目 | 多任务工具箱 (PyPI 包) |
| 训练代码 | 有 | 有 | 无 (仅推理) |
| 任务 | 放置评估 (单点评分) | 放置预测 (全图热力图) | 10+ 子任务 |
| 架构 | ResNet + FC | 双 ResNet + Transformer + UNet | 各模型独立架构 |
| 模型数量 | 2 (SimOPA, SimOPA-ext) | 3 (CNN, ViT, Simple) | 13 个公开 API |
