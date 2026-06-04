# OPA (Object Placement Assessment) 代码架构设计

## 项目概述

OPA 是一个评估合成图像中前景物体放置合理性的深度学习项目。输入一张合成图像及其前景 mask，输出一个二分类评分（0=不合理, 1=合理）。项目包含两个版本的模型：

- **simopa（基线模型）**：仅使用全局图像特征做分类
- **simopa_ext（扩展模型）**：引入前景-背景物体间的空间关系建模，使用 Faster R-CNN 提取参考物体特征，结合注意力机制和几何特征

---

## 1. 代码文件功能说明

### 1.1 根目录 — 基线模型（simopa）

| 文件 | 功能 |
|---|---|
| [config.py](config.py) | 基线模型全局配置。包含数据集路径、图像尺寸(256)、batch size(64)、backbone(resnet18)、训练超参(学习率1e-4, 25 epochs)、实验目录管理等。`Config` 类实例化为全局对象 `opt`，被所有其它模块导入。 |
| [object_place_dataset.py](object_place_dataset.py) | 数据集加载模块。`ImageDataset` 类从 CSV 读取训练/测试集，将 RGB 图像和灰度 mask 分别 resize 到 256×256 后 concat 为 4 通道张量（3通道RGB + 1通道mask）。训练时做随机水平翻转增强。返回 `(img_cat[4,256,256], label, target_box)`。提供 `get_train_dataloader()` 和 `get_test_dataloader()` 工厂函数。 |
| [resnet_4ch.py](resnet_4ch.py) | 改进的 ResNet 实现。核心改动：将 ResNet 的 `conv1` 从 3 通道扩展为 4 通道以接受 RGB+mask 输入。第 4 通道的初始权重由原 RGB 三通道加权平均得到（`0.299R + 0.587G + 0.114B`）。支持 resnet18/34/50/101/152，可从预训练权重加载。还提供 `resnet_for_depth()` 用于深度输入的 2 通道版本。 |
| [object_place_net.py](object_place_net.py) | 基线分类网络。`ObjectPlaceNet` 结构：4 通道 ResNet backbone（去掉最后的池化和 FC 层）→ 1×1 自适应平均池化 → 展平 → 线性分类头（global_feature_dim → 2）。当 `opt.without_mask=True` 时只取前 3 通道，退化为标准 RGB 分类。 |
| [train.py](train.py) | 训练主脚本。流程：初始化模型/损失函数(CrossEntropy)/优化器(Adam)/学习率调度器(MultiStepLR) → 按 epoch 训练 → 在每个 epoch 结束时验证并保存最佳模型（best-acc.pth 和 best-f1.pth）→ 将结果写入 CSV。同时备份源码到实验目录。使用 TensorBoard 记录训练曲线。 |
| [test_model.py](test_model.py) | 测试/评估脚本。加载训练好的模型权重（默认 `best-acc.pth`），在测试集上计算 F1 和 balanced accuracy。可独立运行验证模型性能。 |
| [data_processing/generate_composite.py](data_processing/generate_composite.py) | 合成图像生成工具。通过命令行交互式输入前景 ID、背景 ID、放置位置(x,y,w,h)和 label(0/1)，将前景按 mask 合成到背景上并保存为 JPEG。输出文件名编码了合成参数（fg_id, bg_id, x, y, w, h, scale, label）。 |

### 1.2 根目录 — 扩展模型（simopa_ext）

| 文件 | 功能 |
|---|---|
| [simopa_ext_config.py](simopa_ext_config.py) | 扩展模型配置。继承基线配置并新增：`refer_num=5`（参考物体数量）、`attention_head=16`、`geometric_feature_dim=256`、`binary_mask_size=64`、`roi_align_size=3`、多种关系建模方法（`relation_method`：roi_align / only_target_box / average_all_boxes / without_geometry / simple_geometry / proposed_relation）和多种注意力方法（`attention_method`：only_attention_score / without_attention_score / proposed_attention）。 |
| [simopa_ext_net.py](simopa_ext_net.py) | 扩展分类网络。`ObjectPlaceNet` 在基线基础上增加：**(1) 区域特征提取** — 用 RoI Align 从特征图提取目标/参考物体区域特征；**(2) 几何关系建模** — 用小 CNN 处理目标+参考物体的二值 mask 得到几何特征；**(3) 自注意力聚合** — 对参考物体特征做 self-attention 后加权求和；**(4) 多特征融合** — 将全局特征和聚合后的区域特征 concat 后送入 MLP 分类头。支持 6 种关系方法和 3 种注意力方法。输出为 `(prediction, attention_weights)`。 |

### 1.3 评估/推理脚本

| 文件 | 功能 |
|---|---|
| [eval_opascore/simopa.py](eval_opascore/simopa.py) | 基线模型推理包装器。`ObjectPlacementAssessmentModel` 类：加载基线 `ObjectPlaceNet` 权重，提供 `__call__(image, mask)` 接口，返回 softmax 后的合理性评分（0~1 之间的浮点数）。支持命令行参数指定权重路径、图像路径、mask 路径和 GPU ID。 |
| [eval_opascore/simopa_ext.py](eval_opascore/simopa_ext.py) | 扩展模型推理包装器。`ObjectPlacementAssessmentModel` 类：额外加载 Faster R-CNN 用于提取参考物体特征。`data_preprocess()` 流程：读图像和 mask → 从 mask 计算前景 bbox → 调用 Faster R-CNN 检测背景中的参考物体 → 提取目标/参考物体的 RoI 特征 → 生成二值几何 mask → 送入 simopa_ext 网络。提供二进制 mask 生成工具函数 `generate_binary_mask()` 和 `mask2bbox()`。 |

### 1.4 Faster R-CNN 子模块

| 文件 | 功能 |
|---|---|
| [faster-rcnn/generate_tsv.py](faster-rcnn/generate_tsv.py) | Faster R-CNN 特征提取主入口。提供三个核心函数：`load_model()` 加载预训练的 Faster R-CNN（默认 ResNet-101 backbone，VG 数据集训练）；`get_detections_from_im()` 对单张图像提取 bottom-up attention 特征：运行 RPN → RoI Pooling/Align → 提取所有检测框的 2048 维特征和前景物体的独立特征 → 返回包含 boxes/scores/features/fg_feature 的字典；`generate_tsv()` 批量处理图像并写入 TSV 文件。simopa_ext 直接调用 `get_detections_from_im()` 来获取参考物体特征。 |
| [faster-rcnn/convert_data.py](faster-rcnn/convert_data.py) | TSV 到 numpy 格式转换工具。读取 `generate_tsv.py` 生成的 TSV 文件，将 base64 编码的特征解码为 numpy 数组，按置信度排序后保存为 `.npz` 文件。用于离线预处理特征以加速后续使用。 |
| [faster-rcnn/lib/model/faster_rcnn/faster_rcnn.py](faster-rcnn/lib/model/faster_rcnn/faster_rcnn.py) | Faster R-CNN 核心网络。`_fasterRCNN` 类：RPN 生成候选区域 → RoI Align/Pool 提取区域特征 → 分类头 + 框回归头。特殊设计：`pool_feat=True` 时返回池化后的区域特征（供下游任务使用）；`fgroi` 参数可将前景物体的 RoI 注入候选区域列表，以单独提取前景特征。 |
| [faster-rcnn/lib/model/rpn/rpn.py](faster-rcnn/lib/model/rpn/rpn.py) | RPN（Region Proposal Network）实现。在特征图上滑窗生成 anchors，预测 objectness 分数和框偏移量。 |
| [faster-rcnn/lib/model/rpn/bbox_transform.py](faster-rcnn/lib/model/rpn/bbox_transform.py) | 边界框坐标变换工具。提供 anchors→预测框、预测框→回归目标等坐标转换和裁剪函数。 |
| [faster-rcnn/lib/roi_data_layer/roidb.py](faster-rcnn/lib/roi_data_layer/roidb.py) | ROI 数据库构建。组合多个数据集的标注信息。 |
| [faster-rcnn/lib/roi_data_layer/roibatchLoader.py](faster-rcnn/lib/roi_data_layer/roibatchLoader.py) | 训练时的 ROI 批数据加载器。 |
| [faster-rcnn/lib/model/roi_layers/](faster-rcnn/lib/model/roi_layers/) | RoI 操作层（RoI Pooling、RoI Align、NMS）的 PyTorch 实现。 |
| [faster-rcnn/lib/model/utils/](faster-rcnn/lib/model/utils/) | 工具模块：配置管理(`config.py`)、网络工具(`net_utils.py`)、图像预处理(`blob.py`)、日志(`logger.py`)。 |

---

## 2. 文件间依赖关系

```
config.py  (全局配置)
    │
    ├──► object_place_dataset.py  ← ImageDataset, get_train/test_dataloader
    │       │
    │       └──► train.py  ─────► object_place_net.py  ─────► resnet_4ch.py
    │               │                   │                        │
    │               │                   └── ResNet backbone      └── 4ch conv1 改造
    │               │
    │               └──► test_model.py  ──► object_place_net.py
    │
    └──► eval_opascore/simopa.py  ──► object_place_net.py  (推理)
    
    ─────────────────────────────────────────────────────────

simopa_ext_config.py  (扩展配置)
    │
    └──► simopa_ext_net.py  ──► resnet_4ch.py
            │                       │
            │  ├── roi_align()      └── 4ch conv1 + pretrained weights
            │  ├── SelfAttention
            │  ├── _Bottleneck (几何特征CNN)
            │  └── ObjectPlaceNet (分类 + 关系推理 + 注意力)
            │
            └──► eval_opascore/simopa_ext.py  (扩展推理)
                      │
                      ├──► simopa_ext_net.py
                      └──► faster-rcnn/generate_tsv.py  ← Faster R-CNN 特征提取
                                │
                                ├──► lib/model/faster_rcnn/resnet.py (backbone)
                                ├──► lib/model/faster_rcnn/faster_rcnn.py (核心网络)
                                │       ├──► lib/model/rpn/rpn.py
                                │       ├──► lib/model/roi_layers/ (RoI Align/Pool)
                                │       └──► lib/model/rpn/proposal_target_layer_cascade.py
                                ├──► lib/roi_data_layer/roidb.py
                                └──► lib/model/utils/config.py (Faster R-CNN 配置)
    
    ─────────────────────────────────────────────────────────

data_processing/generate_composite.py  (独立工具，无项目内依赖)

faster-rcnn/convert_data.py  (独立工具，依赖 generate_tsv.py 的输出)
```

关键依赖层级：
1. **config.py** 和 **simopa_ext_config.py** 是整个项目的配置中枢，所有模块均依赖它们提供的 `opt` 全局对象
2. **resnet_4ch.py** 是整个项目的 backbone 基础，两个模型的网络文件都依赖它
3. **faster-rcnn/generate_tsv.py** 在 `simopa_ext.py` 推理时被动态导入，提供参考物体检测和特征提取能力

---

## 3. 训练和推理数据流

### 3.1 训练数据流（simopa / simopa_ext）

```
CSV 文件 (train_set.csv)
  │  columns: ..., label, image_path, mask_path, bbox
  │
  ▼
ImageDataset.__getitem__(index)
  │
  ├── Image.open(image_path) ──► RGB 图像 ──┤
  │                                         ├──► Resize(256,256) ──► ToTensor()
  ├── Image.open(mask_path) ─── 灰度 mask ──┘
  │
  ├── (训练时 50% 概率随机水平翻转 image 和 mask)
  │
  └──► torch.cat([img(3,256,256), mask(1,256,256)], dim=0)
         │
         ▼  img_cat: (B, 4, 256, 256), label: (B,), target_box: (B, 4)
         │
         ▼
  ┌─────────────────────────────────────────────────┐
  │  ObjectPlaceNet.forward(img_cat)                │
  │                                                 │
  │  without_mask? ──► img_cat = img_cat[:,0:3]     │
  │                                                 │
  │  ResNet Backbone (4ch conv1)                    │
  │       │                                         │
  │       ▼  feature_map: (B, C, 8, 8)             │
  │       │                                         │
  │       ├──► AdaptiveAvgPool2d(1) ──► flatten     │
  │       │         │                               │
  │       │         ▼  global_feature: (B, C)       │
  │       │         │                               │
  │       │         ├── (基线) ──► Linear(C,2) ──► logits
  │       │         │                               │
  │       │         └── (扩展) ──► 与区域特征融合 ──► MLP ──► logits
  │       │                               │         │
  │       │                    ┌──────────┘         │
  │       │                    │                    │
  │       │    Faster R-CNN 参考物体特征             │
  │       │    + RoI Align 目标/参考区域特征          │
  │       │    + 几何 mask CNN                       │
  │       │    + Self-Attention 聚合                 │
  │       │                                         │
  └───────┼─────────────────────────────────────────┘
          ▼
  CrossEntropyLoss(logits, label)
          │
          ▼
  Adam optimizer + MultiStepLR scheduler
          │
          ▼
  每 epoch: 验证集评估 → F1 + Balanced Accuracy → 保存最佳模型
```

### 3.2 推理数据流（simopa — 基线模型）

```
输入: composite.jpg, mask.jpg
  │
  ▼
simopa.py: ObjectPlacementAssessmentModel.__call__(image, mask)
  │
  ├── Image.open(image) ──► Resize(256,256) ──► ToTensor() ──► img(3,256,256)
  ├── Image.open(mask)  ──► Resize(256,256) ──► ToTensor() ──► mask(1,256,256)
  │
  └──► cat → (1, 4, 256, 256)
         │
         ▼
  ObjectPlaceNet.forward(img_cat)
         │
         ▼  logits: (1, 2)
         │
  softmax(logits, dim=-1)[0, 1]  ──► score ∈ [0, 1]
```

### 3.3 推理数据流（simopa_ext — 扩展模型）

```
输入: composite.jpg, mask.jpg
  │
  ▼
simopa_ext.py: ObjectPlacementAssessmentModel.__call__(image, mask)
  │
  ├── Image.open(image) ──► RGB ──► Resize(256) ──► ToTensor()
  ├── Image.open(mask)  ──► 灰度 ──► Resize(256) ──► ToTensor()
  │
  ├── mask2bbox(mask) ──► target_box = [x1, y1, x2, y2]
  │
  ├── Faster R-CNN (generate_tsv.get_detections_from_im)
  │     │
  │     ├── 检测背景中所有物体 → boxes + scores + features (2048-d)
  │     ├── 提取前景物体的独立特征 → fg_feature (2048-d)
  │     ├── 按置信度排序取 top-k (refer_num=5) 参考物体
  │     │
  │     └──► refer_box, refer_feat, target_feat
  │
  ├── generate_binary_mask(target_box, refer_box, w, h, 64)
  │     └──► target_mask(1,64,64), refer_mask(5,64,64)
  │
  └──► 所有输入送入 simopa_ext_net.ObjectPlaceNet.forward()
         │
         ├── ResNet Backbone → feature_map (B,512,8,8) → global_feature
         │
         ├── RoI Align 提取目标/参考物体区域特征
         │     └──► target_feature + refer_feature → concat → fc_region_feature
         │
         ├── 几何关系建模 (relation_method=5)
         │     └──► Geometric CNN: concat_mask(2,64,64) → Conv→ReLU→MaxPool×2 → flatten → FC → geometric_feature
         │
         ├── 特征融合: region_feature + geometric_feature → fc_region_feature
         │
         ├── Self-Attention 聚合 (attention_method=2)
         │     └──► region_feature → SelfAttention → similarity_vector
         │          → concat(region_feature, similarity_vector) → fc_weight_learn → softmax
         │          → weighted sum → agg_region_feature
         │
         └── 最终分类
               └──► concat(global_feature, agg_region_feature) → MLP(3层) → logits(1,2)
                      │
                      ▼
               softmax → score ∈ [0, 1]
```

---

## 4. 关键设计决策

### 4.1 4 通道输入
将 mask 作为第 4 通道与 RGB 图像拼接，而非在特征层面融合。这使得可以复用 ImageNet 预训练的 ResNet 权重（通过 RGB→Gray 加权平均将 3 通道权重扩展到 4 通道），同时让网络从浅层就开始感知前景位置信息。

### 4.2 两阶段架构（simopa → simopa_ext）
基线模型仅用全局特征做分类，简洁快速。扩展模型引入 Faster R-CNN 作为固定的参考物体检测器（不参与训练），在 OPA 网络中建模前景-背景物体间的空间和语义关系。两个模型共享同一个 4 通道 ResNet backbone 设计。

### 4.3 Faster R-CNN 的"前景注入"设计
在 `_fasterRCNN.forward()` 中，通过 `fgroi` 参数将前景物体的 bbox 注入 RPN 生成的 RoI 列表，使得 RoI Pooling 能额外提取前景物体特征。这使 simopa_ext 能同时获得参考物体和前景物体的区域特征。

### 4.4 可插拔的关系和注意力方法
`simopa_ext_net.py` 通过 `opt.relation_method`（0-5）和 `opt.attention_method`（0-2）支持多种关系建模和注意力聚合策略的组合，方便进行消融实验。
