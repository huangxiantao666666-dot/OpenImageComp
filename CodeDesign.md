# CodeDesign.md — 图像合成三项目代码详解

## 目录结构总览

```
project/
├── OPA/                              # OPA: 物体放置评估 (训练+推理)
│   ├── config.py                     # 基线模型超参数配置
│   ├── object_place_dataset.py       # PyTorch Dataset + DataLoader
│   ├── object_place_net.py           # 基线模型 ObjectPlaceNet
│   ├── resnet_4ch.py                 # 4通道 ResNet 骨干网络
│   ├── train.py                      # 训练入口 (25 epochs, Adam)
│   ├── test_model.py                 # 测试/评估入口
│   ├── simopa_ext_config.py          # 扩展模型配置
│   ├── simopa_ext_net.py             # 扩展模型 (Faster R-CNN + 自注意力)
│   ├── requirements.txt              # Python 依赖
│   ├── CodeDesign.md                 # 项目自带的架构文档 (中文)
│   ├── eval_opascore/                # 推理封装 (命令行可直接调用)
│   │   ├── simopa.py                 # 基线推理 (加载 SimOPA 权重)
│   │   ├── simopa_ext.py             # 扩展推理 (加载 Faster R-CNN + SimOPA-ext)
│   │   ├── checkpoints/              # 预训练权重存放
│   │   └── examples/                 # 示例图像
│   ├── data_processing/              # 数据合成工具
│   │   └── generate_composite.py     # 交互式合成图像生成
│   └── faster-rcnn/                  # Faster R-CNN 子模块 (特征提取器)
│       ├── generate_tsv.py           # 特征提取主入口
│       ├── convert_data.py           # TSV→numpy 转换
│       └── lib/                      # Faster R-CNN 核心实现
│           ├── model/                # 网络结构 (RPN, RoI, NMS)
│           ├── datasets/             # 数据集处理
│           └── roi_data_layer/       # RoI 数据层
│
├── TopNet-Object-Placement-main/     # TopNet: 物体放置预测 (训练+推理)
│   ├── config.py                     # 配置字典 arg_config
│   ├── train.py                      # 训练脚本 (25 epochs, lr=1e-5)
│   ├── test.py                       # 测试/评估脚本 (F1, bAcc)
│   ├── backbone/                     # 骨干网络
│   │   ├── ResNet.py                 # ResNet18 4ch 骨干 (SOPA 预训练)
│   │   ├── resnet_4ch.py             # 通用 4ch ResNet 变体
│   │   ├── vision_transformer.py     # ViT 完整实现 (timm 风格)
│   │   ├── _builder.py               # 模型构建工厂
│   │   ├── _pretrained.py            # 预训练配置数据类
│   │   ├── _registry.py              # 模型注册表
│   │   └── _manipulate.py            # 模型操作工具
│   ├── network/                      # 网络结构
│   │   ├── network.py                # 主模型: ObPlaNet_resnet18
│   │   ├── network_vit.py            # ViT 变体 (未启用)
│   │   ├── ObPlaNet_simple.py        # FOPA 风格简化版 (动态卷积)
│   │   ├── DynamicModules.py         # 动态滤波器网络 (simpleDFN)
│   │   ├── BaseBlocks.py             # BasicConv2d 定义
│   │   ├── BasicConv2d.py            # Conv+BN+ReLU 块 (重复)
│   │   └── tensor_ops.py             # 张量操作 (upsample_add 等)
│   ├── data/                         # 数据处理
│   │   ├── OBdataset.py              # CPDataset + DataLoader
│   │   └── all_transforms.py         # 联合变换 (JointResize)
│   ├── utils/                        # 工具函数
│   │   ├── metric.py                 # 显著性检测评估指标
│   │   └── misc.py                   # 辅助工具 (AvgMeter, 路径管理)
│   └── best_weight/                  # 检查点保存目录 (空占位)
│
├── libcom-main/                      # libcom: 图像合成工具箱 (纯推理)
│   ├── setup.py                      # PyPI 打包
│   ├── requirements.txt              # Python 依赖
│   ├── libcom/
│   │   ├── __init__.py               # 导出 13 个公开 API
│   │   ├── utils/                    # 共享工具
│   │   │   ├── model_download.py     # HuggingFace/ModelScope 自动下载
│   │   │   ├── environment.py        # GPU 设备检查
│   │   │   └── process_image.py      # 图像读写/BBOX工具
│   │   ├── naive_composition/        # 基础合成 (copy-paste, 高斯, 泊松)
│   │   ├── opa_score/                # ≡ SimOPA (OPA项目的模型)
│   │   ├── fopa_heat_map/            # ≡ FOPA (放置热力图)
│   │   ├── fos_score/                # ≡ DiscoFOS (前景搜索评分)
│   │   ├── harmony_score/            # ≡ BargainNet (和谐度评分)
│   │   ├── image_harmonization/      # PCTNet + LBM (图像协调)
│   │   ├── inharmonious_region_localization/ # MadisNet (不和谐检测)
│   │   ├── color_transfer/           # Reinhard (传统颜色迁移)
│   │   ├── painterly_image_harmonization/    # PHDNet/PHDiffusion
│   │   ├── shadow_generation/        # GPSDiffusion (阴影生成)
│   │   ├── reflection_generation/    # RGDiffusion (反射生成)
│   │   ├── kontext_blending_harmonization/   # FLUX Kontext
│   │   └── os_insert/                # OSInsert (生成式插入)
│   └── docs/                         # 文档 (Sphinx)
│
└── papers/                           # 论文 PDF
    ├── OPA Object Placement Assessment Dataset.pdf
    └── TopNet Transformer-based Object Placement Network for Image Compositing.pdf
```

---

## 一、OPA 项目代码详解

### 1.1 文件依赖关系图

```
config.py ─────────────────────────────────────────────────────────────┐
  │ (定义超参数 opt)                                                     │
  │                                                                     │
  ├──→ object_place_dataset.py ──→ ImageDataset(Dataset)                │
  │      │ 读取 train_set.csv / test_set.csv                            │
  │      │ 加载 RGB + mask, resize 256², concat → [4,256,256]           │
  │      │ get_train_dataloader() / get_test_dataloader()               │
  │      └──→ train.py ──→ 使用 DataLoader 训练                         │
  │              │                                                      │
  ├──→ object_place_net.py ──→ ObjectPlaceNet(nn.Module)               │
  │      │ 4ch ResNet → GAP → Linear(512,2)                             │
  │      │ 依赖 resnet_4ch.py 提供 resnet18(pretrained=True)             │
  │      └──→ train.py ──→ 训练循环 (25 epochs, Adam, MultiStepLR)      │
  │              │ 保存 best-acc.pth / best-f1.pth                       │
  │              │                                                      │
  │              └──→ test_model.py ──→ 加载权重, 评估 F1/bAcc           │
  │                                                                     │
  ├──→ eval_opascore/simopa.py ──→ 独立推理脚本                          │
  │      │ ObjectPlacementAssessmentModel                               │
  │      │ load simopa.pth → 前向 → softmax → score                     │
  │      │ 命令行: python simopa.py --image X --mask Y                   │
  │      │                                                              │
  ├──→ simopa_ext_config.py ──→ 扩展模型超参数                           │
  │      │ refer_num=5, attention_head=16, geometric_feature_dim=256     │
  │      │                                                              │
  ├──→ simopa_ext_net.py ──→ ObjectPlaceNet (扩展版)                     │
  │      │ 4ch ResNet + RoI Align + 自注意力 + 几何特征 + MLP          │
  │      │                                                              │
  │      └──→ eval_opascore/simopa_ext.py ──→ 扩展推理                   │
  │             │ 加载 Faster R-CNN (Visual Genome 预训练)                │
  │             │ 加载 simopa_ext.pth                                    │
  │             │ extract reference features → score                     │
  │                                                                     │
  └──→ resnet_4ch.py ──→ resnet18/34/50/101/152                         │
         │ 标准 ResNet + 4ch conv1 (遮罩通道灰度初始化)                    │
         └──→ 被 object_place_net.py 和 simopa_ext_net.py 复用           │
```

### 1.2 核心文件功能

#### `config.py` — 全局配置
```python
class Config:
    dataset_path = './dataset'
    img_size = 256           # 输入分辨率
    batch_size = 64
    epochs = 25
    backbone = 'resnet18'
    base_lr = 1e-4
    lr_milestones = [10, 16]  # 学习率衰减时间点
    lr_gamma = 0.1
    global_feature_size = 8   # 特征图空间大小 (256/32=8)
```
实例化为全局单例 `opt`，所有模块通过 `from config import opt` 访问。

**设计要点**: 自动创建实验目录 `./experiments/ablation_study/<backbone>/`，包含 `checkpoints/` 和 `logs/` 子目录，避免路径冲突（自动追加 `_repeatN`）。

#### `object_place_dataset.py` — 数据加载
```python
class ImageDataset(Dataset):
    def __getitem__(self, index):
        # 1. 从 CSV 读取路径、标签、边界框
        # 2. PIL 加载 RGB 图像 + 灰度遮罩
        # 3. Resize 均为 256×256
        # 4. 训练时 50% 概率水平翻转 (图像+遮罩同步翻转)
        # 5. Concat: [3,H,W] ⊕ [1,H,W] → [4,H,W]
        # 6. 返回 (img_mask [4,256,256], label, target_box [4])
```

#### `object_place_net.py` — 基线模型
```python
class ObjectPlaceNet(nn.Module):
    def __init__(self):
        # 加载 4ch ResNet (去掉 avgpool 和 fc)
        self.backbone = resnet18(pretrained=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)  # 512=resnet18 最后一层通道数

    def forward(self, img_cat):  # img_cat: [B, 4, 256, 256]
        if opt.without_mask:
            img_cat = img_cat[:, :3]  # 可退化为 3 通道
        feat = self.backbone(img_cat)   # [B, 512, 8, 8]
        feat = self.global_pool(feat)   # [B, 512, 1, 1] → [B, 512]
        logits = self.fc(feat)          # [B, 2]
        return logits
```

#### `resnet_4ch.py` — 4 通道 ResNet
核心技巧在于 **conv1 的第 4 通道权重初始化**:

```python
def resnet18(pretrained=True):
    model = torchvision.models.resnet18(pretrained=True)
    if not opt.without_mask:
        old_weight = model.conv1.weight.data  # [64, 3, 7, 7]
        new_conv1 = nn.Conv2d(4, 64, 7, 2, 3)
        new_weight = new_conv1.weight.data    # [64, 4, 7, 7]
        # 前 3 通道继承预训练权重
        new_weight[:, :3] = old_weight
        # 第 4 通道 = RGB 灰度加权平均
        new_weight[:, 3] = (0.299 * old_weight[:, 0] +
                            0.587 * old_weight[:, 1] +
                            0.114 * old_weight[:, 2])
        model.conv1 = new_conv1
    del model.avgpool, model.fc
    return model
```

**为什么这样初始化**: 遮罩通道是灰度图像，RGB→灰度公式给出了 3 个通道到 1 个通道的最佳线性组合，使第 4 通道有有意义的初始特征提取能力。

#### `train.py` — 训练流程

```
1. 初始化: ObjectPlaceNet(freeze_bn=False)
           + CrossEntropyLoss()
           + Adam(lr=1e-4)
           + MultiStepLR(milestones=[10,16], gamma=0.1)

2. 训练循环 (25 epochs):
   train_epoch():
     for batch in train_loader:
       img_cat [B,4,256,256] → forward → logits [B,2]
       loss = CE(logits, labels)
       loss.backward() → Adam.step()

   test():
     for batch in test_loader:
       forward → compute F1, bAcc

   保存:
     - best-acc.pth (最高 bAcc)
     - best-f1.pth (最高 F1)
     - model-{epoch}.pth (周期保存)

3. 输出: CSV 结果表 + TensorBoard 日志
```

#### `simopa_ext_net.py` — 扩展模型（含空间关系推理）

```python
class ObjectPlaceNet(nn.Module):  # 与基线同名, 但架构更复杂
    def __init__(self):
        self.backbone = resnet18(pretrained=True)  # 4ch ResNet
        self.roi_feature = nn.Linear(2048, 512)     # 区域特征投影
        self.fc_region_feature = nn.Linear(1024, 1024)
        self.geometric_layers = nn.Sequential(
            Conv2d(2, 64, 3, s=2), ReLU, MaxPool(3, s=2),   # → [64, 16, 16]
            Conv2d(64, 256, 3, s=2), ReLU, MaxPool(3, s=2), # → [256, 4, 4]
            Flatten, Linear(4096, 256)                       # → 256
        )
        self.self_attention = SelfAttention(
            dim=1024, heads=16, dim_head=64
        )
        self.fc = nn.Sequential(
            Linear(1536, 1536), ReLU, Dropout(0.1),
            Linear(1536, 512), ReLU,
            Linear(512, 2)
        )

    def forward(self, img_cat, target_box, refer_box,
                target_feat, refer_feat, target_mask, refer_mask):
        # 1. 全局特征
        global_feat = GAP(self.backbone(img_cat))  # [B, 512]

        # 2. 区域特征 (relation_method=5, proposed)
        region_feat = ...  # from Faster R-CNN features
        geo_feat = self.geometric_layers(cat(target_mask, refer_mask))
        region_feat = cat(region_feat, geo_feat)  # [B, N, 1024]

        # 3. 自注意力
        attended_feat, attn_weights = self.self_attention(region_feat)

        # 4. 融合 + 分类
        feat = cat(global_feat, attended_feat)  # [B, 1536]
        logits = self.fc(feat)  # [B, 2]
        return logits, attn_weights
```

**relation_method 枚举**: 共 6 种方法（0~5），默认为 `proposed_relation` (5)。
**attention_method 枚举**: 共 3 种方法（0~2），默认为 `proposed_attention` (2)。

#### `eval_opascore/simopa.py` — 命令行推理

```python
class ObjectPlacementAssessmentModel:
    def __init__(self, device, opt):
        # 加载 ObjectPlaceNet + simopa.pth 权重
    def data_preprocess(self, image, mask):
        # 读取 + resize 256² + 转张量 + concat → [1, 4, 256, 256]
    def __call__(self, image, mask):
        # forward → softmax → return score[0] (合理性概率)

# 命令行:
# python simopa.py --image composite.jpg --mask mask.jpg --weight checkpoints/simopa.pth
```

#### `eval_opascore/simopa_ext.py` — 扩展推理

比基线多两个步骤:
1. **加载 Faster R-CNN**: `build_faster_rcnn()` 从 Visual Genome 预训练权重加载
2. **提取参考特征**: 调用 `get_detections_from_im()` 检测背景物体 → 提取 2048-d 特征 → 取 top-5
3. **生成几何遮罩**: `generate_binary_mask()` 为目标和参考物体生成 64×64 几何遮罩对

```python
class ObjectPlacementAssessmentModel:
    def data_preprocess(self, image, mask):
        h, w = image.size
        target_box = mask2bbox(mask)       # 从遮罩算边界框
        img_cat = concat(image, mask)       # [1, 4, 256, 256]
        # 调用 Faster R-CNN 提取物体特征
        det_results = get_detections_from_im(faster_rcnn, image, target_box)
        target_feat, refer_feat, refer_box = det_results
        target_mask, refer_mask = generate_binary_mask(target_box, refer_box)
        return img_cat, target_box, refer_box, target_feat, refer_feat, ...

    def __call__(self, image, mask):
        # forward → softmax → score
```

#### `data_processing/generate_composite.py` — 数据合成工具

交互式命令行工具，流程:
```
1. 选择前景 ID → 搜索 foreground/ 目录
2. 选择背景 ID → 搜索 background/ 目录
3. 输入位置 (x, y, w, h)
4. Resize 前景到指定大小 → 合成: fg × mask + bg × (1-mask)
5. 输入标签 (0/1)
6. 保存: {fg_id}_{bg_id}_{x}_{y}_{w}_{h}_{scale}_{label}.jpg
```

此脚本**独立运行**，不依赖项目的其他模块。

### 1.3 数据流 (训练)

```
train_set.csv (62,074行)
  │
  ├─ 列: ..., label, ..., composite_path, mask_path, bbox
  │
  ▼
ImageDataset.__getitem__(idx)
  │
  ├─ PIL → RGB Image    → Resize 256² → ToTensor → [3, 256, 256]
  ├─ PIL → Gray Mask    → Resize 256² → ToTensor → [1, 256, 256]
  ├─ Concat → [4, 256, 256]
  │
  ▼
DataLoader(batch=64, shuffle=True)
  │
  ▼
ObjectPlaceNet.forward([B, 4, 256, 256])
  │
  ├─ ResNet backbone (layers conv1~layer4)      → [B, 512, 8, 8]
  ├─ AdaptiveAvgPool2d(1) → flatten              → [B, 512]
  ├─ Linear(512, 2)                              → [B, 2]
  │
  ▼
CrossEntropyLoss(logits, labels) → backward
```

### 1.4 数据流 (SimOPA-ext 推理)

```
composite.jpg + mask.jpg
  │
  ├─→ PIL read + resize 256² + concat → [1, 4, 256, 256]
  │
  ├─→ mask2bbox(mask) → target_box [x1,y1,x2,y2]
  │
  ├─→ Faster R-CNN (VG pretrained):
  │     image → RPN → RoI Align → detect objects
  │     → top-5 detections → each: [class, conf, bbox, 2048-d feat]
  │     → target_feat [2048], refer_feat [5,2048], refer_box [5,4]
  │
  ├─→ generate_binary_mask:
  │     target_box, refer_box → 64×64 rect masks → [5, 2, 64, 64]
  │
  ▼
ObjectPlaceNet (SimOPA-ext).forward(...)
  │
  ├─ Backbone: [1,4,256,256] → [1,512,8,8] → GAP → [1,512]
  ├─ Region: fc(roi_feat) + geometric_layers(binary_mask) → [1,5,1024]
  ├─ SelfAttention: [1,5,1024] → attended [1,1024]
  ├─ Concat: [1,512] ⊕ [1,1024] → [1,1536]
  ├─ MLP: 1536→1536→512→2 → logits [1,2]
  │
  ▼
softmax → score [0] = P(reasonable)
```

---

## 二、TopNet 项目代码详解

### 2.1 文件依赖关系图

```
config.py ──────────────────────────────────────────────────────────────┐
  │ arg_config = { model='ObPlaNet_resnet18',                         │
  │   epoch_num=25, lr=1e-5, batch_size=8, input_size=256 ... }       │
  │                                                                     │
  ├──→ data/OBdataset.py ──→ CPDataset(Dataset)                        │
  │      │ 读取 JSON 标注 (train_pair_new.json)                         │
  │      │ 加载 bg + fg + mask, JointResize 256²                        │
  │      │ 生成逐像素标签图 (pos=1, neg=0, unlabeled=255)               │
  │      │ create_loader() 返回 DataLoader                              │
  │      │                                                              │
  │      ├──→ data/all_transforms.py                                    │
  │      │      JointResize(bg,fbo,mask) → 同步 resize                  │
  │      │      Compose, Compose_heatmap                                │
  │      │                                                              │
  │      └──→ train.py                                                 │
  │             │ Trainer 类                                            │
  │             │ train_outs = net(bg, fg, mask, 'train')               │
  │             │ loss = CE(logits, target, ignore_index=255)           │
  │             │                                                       │
  ├──→ network/network.py ──→ ObPlaNet_resnet18(nn.Module)             │
  │      │ 主模型: 双向编码器 + Transformer + UNet 解码器               │
  │      │                                                              │
  │      ├──→ backbone/ResNet.py                                        │
  │      │      Backbone_ResNet18_in3 (bg encoder, SOPA预训练)          │
  │      │      Backbone_ResNet18_in3_1 (fg encoder, ImageNet预训练)    │
  │      │      pretrained_resnet18_4ch()                               │
  │      │                                                              │
  │      ├──→ network/BaseBlocks.py                                     │
  │      │      BasicConv2d(Conv2d→BN→ReLU)                             │
  │      │                                                              │
  │      ├──→ network/tensor_ops.py                                     │
  │      │      cus_sample, upsample_add, upsample_cat                  │
  │      │                                                              │
  │      └──→ network/ObPlaNet_simple.py (变体)                         │
  │             ├──→ network/DynamicModules.py                          │
  │             │      simpleDFN (动态卷积融合)                          │
  │             └──→ backbone/ResNet.py (共享编码器)                     │
  │                                                                     │
  ├──→ network/network_vit.py ──→ ObPlaNet_vit (备用模型)              │
  │      │ 使用 ViT-Base/16 代替 ResNet18 作为编码器                   │
  │      │ 当前未启用 (network/__init__.py 中注释)                       │
  │      │                                                              │
  │      ├──→ backbone/vision_transformer.py                            │
  │      │      完整的 ViT 实现 (timm 风格, 支持各种预训练)              │
  │      │                                                              │
  │      ├──→ backbone/_builder.py                                      │
  │      │      build_model_with_cfg() 工厂函数                         │
  │      │                                                              │
  │      └──→ backbone/_pretrained.py                                   │
  │             PretrainedCfg 数据类                                    │
  │                                                                     │
  └──→ test.py ──→ 测试/评估                                            │
         │ 加载检查点 → eval() → 逐像素预测                             │
         │ 计算 TP/TN/FP/FN → F1 分数 + bAcc                           │
         │                                                              │
         └──→ utils/metric.py ──→ CalFM, CalMAE, CalSM, CalEM, CalWFM  │
                (显著性检测评估指标，实际测试未使用)                      │
```

### 2.2 核心文件功能

#### `config.py` — 配置字典

```python
arg_config = {
    'model': 'ObPlaNet_resnet18',
    'epoch_num': 25,          # 训练轮次
    'lr': 1e-5,               # 学习率
    'batch_size': 8,          # 批次大小 (受显存限制)
    'input_size': 256,        # 输入分辨率
    'optim': 'Adam_trick',    # 优化器选择
    'lr_type': 'all_decay',   # 学习率策略: lr * 0.5^(epoch//2)
    'weight_decay': 0.0001,
    'checkpoint_dir': './best_weight',
    'suffix': 'simple_mask_adam',  # 实验名后缀
}
```

#### `data/OBdataset.py` — 数据集

```python
class CPDataset(Dataset):
    def __init__(self, json_path, bg_dir, fg_dir, mask_dir, transforms):
        # 解析 JSON → self.samples (list of dicts)
        # 每条样本包含: annID, scID, pos_label, neg_label, scale, ...

    def _obtain_target(self, pos_label, neg_label):
        # 初始化 target [256,256] = 255 (全部忽略)
        # pos_label 坐标 → target[y, x] = 1
        # neg_label 坐标 → target[y, x] = 0
        # 坐标放缩: pos_label 是原图坐标，需要按比例映射到 256²

    def __getitem__(self, index):
        # 1. 加载 bg, fg (3ch), mask (1ch)
        # 2. JointResize → 256²
        # 3. 随机水平翻转 (bg, fg, mask, target 同步)
        # 4. make_composite: fg × mask + bg × (1-mask) → [B, 4, 256, 256]
        # 5. 返回: (index, bg, mask, fg, target, num_labels, ...)

def make_composite_PIL(bg, fg, mask, bbox, ...):
    """推理时使用: 将前景放到背景的指定 bbox 位置"""
    fg_resized = fg.resize(bbox_size)
    mask_resized = mask.resize(bbox_size)
    fg_pasted = Image.new('RGBA', bg.size)
    fg_pasted.paste(fg_resized, bbox)
    # Alpha 混合
    composite = fg × mask + bg × (1-mask)
```

#### `network/network.py` — 主模型 ObPlaNet_resnet18

```python
class ObPlaNet_resnet18(nn.Module):
    def __init__(self):
        # 编码器
        self.bg_encoder = Backbone_ResNet18_in3()   # → 5 层特征 (div2~div32)
        self.fg_encoder = Backbone_ResNet18_in3_1() # → 6 层特征 (div1~div32)

        # Transformer (将 bg 和 fg 的特征拼接后做全局建模)
        self.transformer = Transformer(
            input_dim=1024,       # 512(bg) + 512(fg)
            hidden_dim=128,       # MLP 中间层
            num_heads=8,
            num_layers=4
        )

        # 解码器 (UNet 风格上采样 + 跳连)
        self.upconv32 = BasicConv2d(1024, 512)
        self.upconv16 = BasicConv2d(512, 256)
        self.upconv8  = BasicConv2d(256, 128)
        self.upconv4  = BasicConv2d(128, 64)
        self.upconv2  = BasicConv2d(64, 64)
        self.upconv1  = BasicConv2d(64, 64)

        # 分类头
        self.deconv = nn.ConvTranspose2d(64, 512, 3, 1, 1)
        self.classifier = nn.Conv2d(512, 2, 1)

    def forward(self, bg, fg, mask, mode='train'):
        # 构建编码器输入
        emask = torch.zeros_like(mask)  # 空遮罩 (背景编码器用)
        fg_input = torch.cat([fg, mask], dim=1)   # [B, 4, 256, 256]
        bg_input = torch.cat([bg, emask], dim=1)  # [B, 4, 256, 256]

        # 编码
        bg_feats = self.bg_encoder(bg_input)
        fg_feats = self.fg_encoder(fg_input)

        # 拼接最深特征
        x = torch.cat([bg_feats['div32'], fg_feats['div32']], dim=1)
        # [B, 1024, 8, 8]

        # Transformer 全局建模
        x = self.transformer(x)  # [B, 1024, 8, 8]

        # UNet 解码 + 跳连
        x = self.upconv32(x)                          # [B, 512, 8, 8]
        x = upconv_add(x, bg_feats['div16'])          # [B, 256, 16, 16]
        x = upconv_add(x, bg_feats['div8'])           # [B, 128, 32, 32]
        x = upconv_add(x, bg_feats['div4'])           # [B, 64, 64, 64]
        x = upconv_add(x, bg_feats['div2'])           # [B, 64, 128, 128]
        x = upconv(x, scale=2)                        # [B, 64, 256, 256]

        # 分类
        x = self.deconv(x)      # [B, 512, 256, 256]
        out = self.classifier(x) # [B, 2, 256, 256]
        return out, x
```

**Transformer 内部**:

```python
class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        n_tokens = 64  # 8×8 特征图展平后的 token 数
        self.layers = nn.ModuleList([
            TransformerLayer(input_dim, hidden_dim, num_heads, n_tokens)
            for _ in range(num_layers)
        ])

class TransformerLayer(nn.Module):
    def forward(self, x):
        # x: [B, dim, 8, 8]
        B, C, H, W = x.shape
        tokens = x.view(B, C, -1).permute(2, 0, 1)  # [64, B, 1024]
        # MultiheadAttention + Residual
        attn_out = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        # MLP + Residual
        mlp_out = self.mlp(tokens.permute(1, 0, 2).flatten(1))
                    .view(B, H*W, C).permute(1, 0, 2)
        tokens = self.norm2(tokens + mlp_out)
        return tokens.permute(1, 2, 0).view(B, C, H, W)
```

#### `backbone/ResNet.py` — 双编码器

```python
def Backbone_ResNet18_in3(pretrained=True):
    """背景编码器: 使用 SOPA 预训练的 4ch ResNet18"""
    model = pretrained_resnet18_4ch()
    if pretrained:
        state_dict = torch.load('./data/data/SOPA.pth.tar')
        model.load_state_dict(state_dict)
    # 拆分为 5 个子模块 → {div_2, div_4, div_8, div_16, div_32}
    return nn.ModuleDict({...})

def Backbone_ResNet18_in3_1(pretrained=True):
    """前景编码器: ImageNet 预训练 + 第4通道灰度初始化"""
    model = torchvision.models.resnet18(pretrained=True)
    old_weight = model.conv1.weight.data  # [64, 3, 7, 7]
    new_conv1 = nn.Conv2d(4, 64, 7, 2, 3)
    new_conv1.weight.data[:, :3] = old_weight
    new_conv1.weight.data[:, 3] = (0.299*old_weight[:,0] +
                                    0.587*old_weight[:,1] +
                                    0.114*old_weight[:,2])
    model.conv1 = new_conv1
    # 拆分为 6 个子模块 → {div_1, div_2, div_4, div_8, div_16, div_32}
```

#### `train.py` — 训练流程

```python
class Trainer:
    def __init__(self, args):
        self.net = ObPlaNet_resnet18()  # 或 ObPlaNet_vit()
        self.optimizer = Adam(self.net.parameters(), lr=1e-5)
        self.criterion = CrossEntropyLoss(ignore_index=255)

    def train_epoch(self, loader, epoch):
        for batch in loader:
            # batch: (idx, bg, mask, fg, target, ...)
            bg, fg, mask, target = batch[1:5]
            # bg: [B,3,256,256], fg: [B,3,256,256]
            # mask: [B,1,256,256], target: [B,256,256]

            loss = self._compute_loss(bg, fg, mask, target)
            loss.backward()
            optimizer.step()

    def _compute_loss(self, bg, fg, mask, target):
        outs, _ = self.net(bg, fg, mask, 'train')
        # outs: [B, 2, 256, 256] → logits
        # target: [B, 256, 256] → values in {0, 1, 255}
        return self.criterion(outs, target)
```

学习率调度:
```python
# all_decay: lr = initial_lr * (0.5 ** (epoch // 2))
# epoch 0-1: lr=1e-5
# epoch 2-3: lr=5e-6
# epoch 4-5: lr=2.5e-6
# ...
```

#### `test.py` — 评估流程

```python
def evaluate(load_path):
    net = ObPlaNet_resnet18()
    net.load_state_dict(torch.load(load_path))
    net.eval()

    TP = TN = FP = FN = 0
    for batch in test_loader:
        outs, _ = net(bg, fg, mask, 'val')
        preds = outs.argmax(dim=1)  # [B, 256, 256]
        # 仅统计 target != 255 的像素
        TP += ((preds == 1) & (target == 1)).sum()
        TN += ((preds == 0) & (target == 0)).sum()
        FP += ((preds == 1) & (target == 0)).sum()
        FN += ((preds == 0) & (target == 1)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    bAcc = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
```

#### `network/DynamicModules.py` — 动态卷积融合

```python
class simpleDFN(nn.Module):
    """从前景特征生成动态卷积核, 应用于背景特征"""
    def __init__(self, in_channels, out_channels):
        self.fuse = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, y):
        # x: 背景特征 [B, C_bg, H, W]
        # y: 前景特征 [B, C_fg, H, W]
        kernel = self.gernerate_kernel(y)  # 从前景生成核
        # Unfold 背景 → 分组卷积
        x_unfolded = F.unfold(x, kernel_size=3, padding=1)
        x_filtered = F.conv2d(x_unfolded, kernel, groups=B*C)
        return self.fuse(x_filtered)
```

#### `network/ObPlaNet_simple.py` — FOPA 风格简化版

```python
class ObPlaNet_resnet18(nn.Module):
    def __init__(self):
        self.bg_encoder = Backbone_ResNet18_in3()    # 冻结参数
        self.fg_encoder = Backbone_ResNet18_in3_1()
        # 动态卷积模块 (代替 Transformer)
        self.selfdc_16 = simpleDFN(512, 64)  # 在 8×8 分辨率融合
        self.selfdc_8 = simpleDFN(256, 64)   # 在 16×16 分辨率融合
        # 简化解码器
        # ...

    def forward(self, bg, fg, mask, mode):
        bg_feats = self.bg_encoder(bg, emask)
        fg_feats = self.fg_encoder(fg, mask)
        # 动态卷积融合 (无 Transformer)
        x16 = self.selfdc_16(bg_feats['div32'], fg_feats['div32'])
        x8  = self.selfdc_8(bg_feats['div16'], fg_feats['div16'])
        # ...
```

### 2.3 数据流 (训练)

```
train_pair_new.json (标注)
  │
  │ 每条记录: {annID, scID, pos_label: [(x,y),...], neg_label: [...], scale, ...}
  │
  ▼
CPDataset.__getitem__(idx)
  │
  ├─ 加载 bg.jpg (3ch), fg.jpg (3ch), mask.jpg (1ch)
  ├─ JointResize → 256²
  ├─ RandomHorizontalFlip (同步: bg, fg, mask, target)
  ├─ 生成 target:
  │   - 初始化 [256,256] = 255
  │   - pos_label 坐标 → target[y,x] = 1  (合理位置)
  │   - neg_label 坐标 → target[y,x] = 0  (不合理位置)
  │   - 剩余像素 = 255 (ignore)
  │
  ▼
DataLoader(batch=8, shuffle=True)
  │
  ▼
ObPlaNet_resnet18.forward(bg [8,3,256,256], fg [8,3,256,256], mask [8,1,256,256])
  │
  ├─ bg_encoder(bg ⊕ zero_mask [4ch]) → 5层特征图
  ├─ fg_encoder(fg ⊕ mask [4ch])      → 6层特征图
  ├─ cat(bg_div32 [8,512,8,8], fg_div32 [8,512,8,8]) → [8,1024,8,8]
  ├─ Transformer (4层, 8头自注意力, 64 tokens)       → [8,1024,8,8]
  ├─ UNet Decoder (5级上采样 + 跳连)                   → [8,64,256,256]
  ├─ ConvTranspose(64→512) + Conv(512→2)              → [8,2,256,256]
  │
  ▼
CrossEntropyLoss(logits=[8,2,256,256], target=[8,256,256], ignore=255)
  → backward → Adam.step
```

### 2.4 数据流 (推理)

```
bg.jpg + fg.jpg + mask.jpg
  │
  ├─ 多个尺度 (16个, 1/18 ~ 16/18):
  │   prepare_multi_fg_scales() → 生成多尺度前景
  │
  ▼
对每个尺度:
  ObPlaNet_resnet18.forward() → [2, 256, 256] 热力图
  → softmax → 取通道1 (合理类) 的概率值
  │
  ▼
跨尺度堆叠 heatmaps → 取 top-50 最高分位置
  │
  ▼
make_composite_PIL 生成合成图
  │
  ▼
输出: box_list (最佳放置 bbox) + composite_list
```

---

## 三、libcom 项目代码详解

### 3.1 整体架构设计模式

libcom 的代码组织遵循统一的设计模式:

```
每个子功能包:
  libcom/<task_name>/
    ├── __init__.py              → 导出任务入口类
    ├── <task>_prediction.py     → 主类: 封装预处理+推理+后处理
    ├── source/                  → 模型定义 + 配置 + 工具
    │   ├── __init__.py
    │   ├── <model>.py           → 网络结构定义
    │   ├── config/              → YAML 配置
    │   └── backbone/            → (可选) 骨干网络
    └── checkpoints/             → 预训练权重 (自动下载)
```

**统一使用模式**:
```python
from libcom import SomeModel
model = SomeModel(device='cuda:0')
result = model(input1, input2, ...)
```

### 3.2 依赖关系总图

```
libcom/__init__.py ──→ 导出 13 个公共 API
  │
  ├── utils/model_download.py
  │    ├─ download_pretrained_model(): hf_hub_download → model_file_download
  │    │   ├─ HuggingFace: BCMIZB/Libcom_pretrained_models
  │    │   └─ ModelScope: yujieouo/Libcom_pretrained_models
  │    └─ download_entire_folder(): 下载 ZIP → 解压
  │    (被所有 Model 类调用, 首次使用时自动下载权重)
  │
  ├── utils/process_image.py
  │    ├─ read_image_pil(), read_image_opencv()
  │    ├─ pil_to_cv2(), cv2_to_pil()
  │    ├─ mask_to_bbox()
  │    └─ make_image_grid(), draw_bbox_on_image()
  │    (被所有 Model 类使用, 用于图像预处理)
  │
  ├── utils/environment.py
  │    └─ check_gpu_device() → 验证 GPU 可用性
  │
  ├── naive_composition/ ──→ get_composite_image()
  │    ├─ 直接复制粘贴: fg.resize(bbox) → paste
  │    ├─ 高斯模糊混合: GaussianBlur(mask) → alpha blend
  │    └─ 泊松融合: cv2.seamlessClone(NORMAL_CLONE)
  │
  ├── color_transfer/ ──→ color_transfer()
  │    └─ 纯 NumPy/CV2 实现, 无深度学习
  │       BGR → Lab → 统计对齐 → BGR
  │
  ├── opa_score/ ──→ OPAScoreModel
  │    ├─ object_place_net.py: ObjectPlaceNet (4ch ResNet + FC)
  │    │   ├─ resnet_4ch.py: 4ch ResNet (与 OPA 项目相同)
  │    │   └─ oblect_place_config.py: 配置
  │    └─ 权重: SimOPA.pth
  │
  ├── fopa_heat_map/ ──→ FOPAHeatMapModel
  │    ├─ ObPlaNet_simple.py: 双流编码器 + 动态卷积融合
  │    │   ├─ backbone/ResNet.py: 4ch ResNet (SOPA 预训练)
  │    │   ├─ DynamicModules.py: simpleDFN
  │    │   └─ BaseBlocks.py: BasicConv2d
  │    ├─ prepare_multi_fg_scales.py: 多尺度前景生成
  │    ├─ OBdataset.py: make_composite_PIL
  │    └─ 权重: FOPA.pth, SOPA.pth
  │
  ├── fos_score/ ──→ FOSScoreModel
  │    ├─ networks.py: SingleScaleD (VGG19 判别器) + StudentModel (蒸馏)
  │    ├─ loss_function.py: TripletLoss, GANLoss, KL
  │    ├─ config/config_rfosd.yaml / config_sfosd.yaml
  │    └─ 权重: FOS_D.pth
  │
  ├── harmony_score/ ──→ HarmonyScoreModel
  │    ├─ bargainnet.py: StyleEncoder (5层 PartialConv → 16-d 风格向量)
  │    └─ 权重: BargainNet.pth
  │
  ├── image_harmonization/ ──→ ImageHarmonizationModel
  │    ├─ pct_net.py: PCTNet (ViT + Per-Pixel Color Transform)
  │    │   ├─ ViT_Harmonizer (patch embed + 9层Transformer)
  │    │   └─ PCT (线性/多项式/二次/cubic 颜色变换)
  │    ├─ functions.py: ViT_Harmonizer, PCT, 3D LUT, TrilinearInterp
  │    ├─ source/src/lbm/: LBM 扩散模型 (VAE + FlowMatch)
  │    └─ 权重: PCTNet.pth, IdentityLUT33.txt, lbm_ckpt/
  │
  ├── inharmonious_region_localization/ ──→ InharmoniousLocalizationModel
  │    ├─ madis_net.py: MadisNet (DIRLNet + HDRPointwiseNN)
  │    │   ├─ DIRLNet: ResNet-Encoder + Dual Attention + Decoder
  │    │   └─ HDRPointwiseNN: 双边网格颜色重调
  │    ├─ blocks.py: BasicBlock, Bottleneck, Conv2d_cd, PartialConv2d
  │    └─ 权重: Inharmonious_G.pth, IHDRNet.pth
  │
  ├── shadow_generation/ ──→ ShadowGenerationModel
  │    └─ GPSDiffusion: ControlNet + UNet2D + DDPM
  │       权重: Shadow_cldm.ckpt, Shadow_ppp.ckpt, Shadow_reg.pth
  │
  ├── reflection_generation/ ──→ ReflectionGenerationModel
  │    └─ RGDiffusion: ControlNet + Diffusion
  │       权重: Reflection_cldm.ckpt, Reflection_ppp.ckpt, Reflection_reg.pth
  │
  ├── painterly_image_harmonization/ ──→ PainterlyHarmonizationModel
  │    ├─ PHDNet: 专用绘画协调网络
  │    └─ PHDiffusion: 扩散模型 + Adapter
  │
  ├── kontext_blending_harmonization/ ──→ KontextBlendingHarmonizationModel
  │    └─ FLUX.1-Kontext-dev + LoRA (黑色森林实验室)
  │       权重: flux_kontext_blending.safetensors, flux_kontext_harmonization.safetensors
  │
  └── os_insert/ ──→ OSInsertModel
       ├─ aggressive 模式: ObjectStitch + SAM + InsertAnything
       └─ conservative 模式: 背景+bbox → mask → FLUX Fill 插入
```

### 3.3 核心设计模式: 自动权重下载

```python
# libcom/utils/model_download.py (每个 Model 类在 __init__ 中调用)

def download_pretrained_model(weight_path: str) -> None:
    """若权重文件不存在, 自动从 HuggingFace 或 ModelScope 下载"""
    if os.path.exists(weight_path):
        return  # 已存在, 跳过

    weight_name = os.path.basename(weight_path)

    # 尝试 HuggingFace (主源)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id='BCMIZB/Libcom_pretrained_models',
            filename=weight_name,
            local_dir=os.path.dirname(weight_path),
            local_dir_use_symlinks=False
        )
        return
    except Exception:
        pass  # 回退到 ModelScope

    # 回退: ModelScope (中国境内更快)
    from modelscope.hub.file_download import model_file_download
    model_file_download(
        model_id='yujieouo/Libcom_pretrained_models',
        file_path=weight_name,
        cache_dir=os.path.dirname(weight_path)
    )
```

### 3.4 各模块数据流

#### OPAScoreModel

```
composite_image (str/PIL/ndarray)
composite_mask  (str/PIL/ndarray)
  │
  ├─ preprocess:
  │   read + resize 256² + concat → [1, 4, 256, 256]
  │
  ▼
ObjectPlaceNet (4ch ResNet18 → GAP → Linear(512,2))
  │
  ▼
softmax → score[1] = P(reasonable)
  │
  ▼
float ∈ [0, 1]
```

#### FOPAHeatMapModel

```
background_image + foreground_image + foreground_mask
  │
  ├─ prepare_multi_fg_scales():
  │   生成 16 个尺度 (1/18 ~ 16/18 of background)
  │   每个尺度: resize fg → center on bg canvas → save
  │
  ▼
For each scale:
  ObPlaNet_resnet18.forward(bg, fg_scaled, mask_scaled)
  → 热力图 [2, 256, 256]
  → softmax → 取合理通道
  │
  ▼
跨尺度 stack → 取 top-50 最高分像素位置
  │
  ▼
make_composite_PIL(bg, fg, mask, top_bbox)
  │
  ▼
box_list (最佳放置位置), heatmap_list (热力图路径)
```

#### HarmonyScoreModel

```
composite_image + composite_mask
  │
  ├─ resize 256²
  │
  ▼
StyleEncoder.forward(image, mask=fg_mask)    → bg_style [16]
StyleEncoder.forward(image, mask=bg_mask)    → fg_style [16]
  │  (PartialConv 跳过被 mask 掉的区域)
  │
  ▼
score = exp(-0.04212 × ||bg_style - fg_style||₂)
  │
  ▼
float ∈ [0, 1]
```

#### ImageHarmonizationModel (PCTNet 模式)

```
composite_image + composite_mask
  │
  ├─ low-res branch: resize 256²
  │   │
  │   ├─ ViT Encoder:
  │   │   patch_embedding(image ⊕ mask) → 256 tokens (16×16)
  │   │   + 9层 Transformer (2 heads, dim=256, GELU)
  │   │
  │   ├─ ViT Decoder:
  │   │   ConvTranspose2d → per-pixel parameter map [12, 256, 256]
  │   │
  │   └─ PCT Module:
  │       color_params[i,j] = 3×3 matrix + 3-d bias (线性模式)
  │       output[i,j] = input[i,j] × M₃ₓ₃ + bias₃
  │
  ├─ full-res branch: bicubic interpolate param map → apply to full-res image
  │
  ▼
output = transformed_fg × mask + original_bg × (1 - mask)
```

---

## 四、跨项目代码依赖关系

### 4.1 共享的技术组件

| 组件 | OPA | TopNet | libcom |
|------|:---:|:------:|:------:|
| 4 通道 ResNet18 | ✅ 核心 | ✅ 核心 | ✅ 复用 |
| 遮罩通道灰度初始化 (0.299R+0.587G+0.114B) | ✅ resnet_4ch.py | ✅ backbone/ResNet.py | ✅ opa_score/resnet_4ch.py |
| SOPA 预训练权重 | ✅ 基线权重 | ✅ bg_encoder 权重 | ✅ FOPA 编码器权重 |
| Alpha 混合合成 | ✅ generate_composite.py | ✅ OBdataset.py | ✅ naive_composition |
| 动态卷积 (simpleDFN) | ❌ | ✅ ObPlaNet_simple | ✅ FOPA 编码器 |
| Transformer 融合 | ❌ | ✅ 核心创新 | ❌ |
| Faster R-CNN 特征提取 | ✅ SimOPA-ext | ❌ | ❌ |
| 自注意力 + 几何特征 | ✅ SimOPA-ext | ❌ | ❌ |
| Partial Convolution | ❌ | ❌ | ✅ BargainNet, MadisNet |
| 扩散模型 (Diffusion) | ❌ | ❌ | ✅ LBM, Shadow, Reflection |
| ViT 编码器 | ❌ | ✅ (备用) | ✅ PCTNet |
| Reinhard 颜色迁移 | ❌ | ❌ | ✅ color_transfer |

### 4.2 进化谱系

```
SOPA (CVPR 2022) ──────────────────────────┐
  │ 4ch ResNet + FC                         │
  │                                         │
  ├──→ SimOPA (OPA 基线)                     │
  │      simopa.pth ← 直接使用 SOPA 权重     │
  │      └──→ libcom/opa_score/              │
  │             OPAScoreModel 等价于 SimOPA  │
  │                                         │
  ├──→ FOPA (ECCV 2022)                     │
  │      双流编码器 + 动态卷积融合            │
  │      └──→ libcom/fopa_heat_map/          │
  │             FOPAHeatMapModel 等价于 FOPA │
  │             └──→ TopNet 使用 SOPA 作为    │
  │                  背景编码器预训练权重      │
  │                                         │
  └──→ TopNet (CVPR 2023)                   │
        SOPA → bg_encoder 预训练             │
        + Transformer + UNet Decoder        │
        + ViT 变体 (备用)                    │
        └──→ 评估时对比 SOPA 和 FOPA         │

SimOPA ──→ SimOPA-ext (OPA 扩展)            │
  + Faster R-CNN (Visual Genome 预训练)      │
  + Self-Attention + Geometric Features     │
```

### 4.3 关键设计决策对比

| 决策 | OPA | TopNet | libcom |
|------|-----|--------|--------|
| **输入格式** | RGB+Mask 拼接 (4ch) | 分别编码 BG 和 FG | 因任务而异 |
| **骨干网络** | ResNet18/34/50 | ResNet18 (固定) | 各模型独立 |
| **是否冻结编码器** | 不冻结 | bg 编码器有时冻结 | 按任务配置 |
| **特征融合** | 无 (仅全局池化) | Transformer 4层 | 动态卷积/自注意力/拼接 |
| **输出粒度** | 整图评分 (1值) | 逐像素评分图 (256²) | 因任务而异 |
| **评估指标** | F1 + bAcc | F1 + bAcc | 各任务独立 |
| **数据集拆分** | 训练/前景/背景严格不重叠 | 同 SOPA 协议 | 各任务独立 |
| **预训练依赖** | ImageNet | ImageNet + SOPA | 各模型独立下载 |

### 4.4 文件间直接调用关系

```
OPA 项目内部:
  train.py → config.py, object_place_dataset.py, object_place_net.py, resnet_4ch.py
  test_model.py → config.py, object_place_dataset.py, object_place_net.py
  simopa.py → object_place_net.py, config.py
  simopa_ext.py → simopa_ext_net.py, simopa_ext_config.py, faster-rcnn/
  generate_composite.py → (独立, 无项目内依赖)

TopNet 项目内部:
  train.py → config.py, data/OBdataset.py, network/network.py
  test.py → config.py, data/OBdataset.py, network/network.py
  network/network.py → backbone/ResNet.py, network/BaseBlocks.py, network/tensor_ops.py
  network/network_vit.py → backbone/vision_transformer.py
  network/ObPlaNet_simple.py → network/DynamicModules.py, backbone/ResNet.py
  backbone/ResNet.py → backbone/resnet_4ch.py (部分复用)

libcom 内部:
  libcom/__init__.py → 所有 13 个 Model 类
  每个 Model 类 → utils/model_download.py (自动下载权重)
  每个 Model 类 → utils/process_image.py (图像预处理)
  naive_composition → (无项目内深度学习依赖)
  color_transfer → (纯 NumPy, 无项目内依赖)
  opa_score → (自包含, 有独立 resnet_4ch.py)
  fopa_heat_map → (自包含, 有独立 backbone/ResNet.py)

跨项目:
  TopNet → 加载 SOPA 预训练权重 (来自 OPA/SOPA 项目)
  libcom/opa_score → 复用 OPA 项目的 SimOPA 架构和权重
  libcom/fopa_heat_map → 复用 FOPA 的项目架构和权重
```
