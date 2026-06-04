# 方向A 具体实现方案

## 0. 要新建的项目目录

```
project/placement_app/                  # 我们新建的应用目录
├── models/
│   ├── __init__.py
│   ├── resnet_4ch.py                  # 【改】从 OPA 复制 + 加 base_width
│   ├── simopa.py                      # 【改】从 OPA 解耦 + 加载我们自己的模型
│   └── weights/
│       └── simopa.pth                 # 【拷】从 OPA/eval_opascore/checkpoints/ 复制
├── pipeline/
│   ├── __init__.py
│   ├── candidates.py                  # 【新】候选位置生成
│   ├── composite.py                   # 【新】合成图像
│   └── scorer.py                      # 【新】评分流水线
├── app.py                             # 【新】Gradio 应用主入口
├── grad_cam.py                        # 【新】Grad-CAM 解释 (进阶项)
├── test_cases.py                      # 【新】测试案例脚本
├── assets/
│   ├── backgrounds/                   # 预设背景素材
│   └── foregrounds/                   # 预设前景素材 (带透明背景的PNG)
└── requirements.txt                   # 【新】
```

---

## 第一步：解耦模型——去掉 `config.py` 的全局依赖

**问题**：`OPA/resnet_4ch.py` 依赖 `from config import opt`，`opt` 是模块级全局单例。这意味着同一进程不能加载两个不同配置的模型（比如我们想同时比较原版和轻量版）。

**做法**：把 `resnet_4ch.py` 和 `object_place_net.py` 中所有 `opt.xxx` 替换为构造函数参数。

### 文件 1: `models/resnet_4ch.py`

从 `OPA/resnet_4ch.py` 复制，做以下改动：

**改动点 1**：`resnet()` 函数签名从
```python
def resnet(layers, pretrained=False, pretrained_weight=None, **kwargs):
```
改为
```python
def resnet(layers, pretrained=False, pretrained_weight=None,
           base_width=64, without_mask=False, in_channels=4, **kwargs):
```

**改动点 2**：`ResNet.__init__` 中所有硬编码的通道数改为 `base_width` 的倍数：
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, base_width=64):
        self.inplanes = base_width
        super(ResNet, self).__init__()
        
        # conv1: 用 base_width 替代 64
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个 stage: 原来 [64,128,256,512] → [w, 2w, 4w, 8w]
        self.layer1 = self._make_layer(block, base_width,      layers[0])
        self.layer2 = self._make_layer(block, base_width * 2,  layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4,  layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8,  layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)
```

**改动点 3**：`resnet()` 函数中，`conv1` 输出通道从 64 改为 `base_width`：
```python
# 原来: model.conv1 = nn.Conv2d(4, 64, kernel_size=7, ...)
# 改为:
model.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2,
                         padding=3, bias=False)
```

**改动点 4**：权重加载适配。加载 ImageNet 预训练权重时，如果 `base_width != 64`，只加载形状匹配的层（conv1 和 fc 跳过），其余层正常加载：
```python
if pretrained:
    pretrained_state_dict = torch.load(pretrained_weight)
    if base_width != 64:
        # 轻量模型: 跳过不匹配的 conv1.weight 和 fc.weight, fc.bias
        model_state = model.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state)
        print(f'Loaded partial pretrained weights (base_width={base_width})')
    else:
        # 正常加载 (原版)
        ...
```

**结果**：
```python
# 原版: ~11.2M 参数
model_full = resnet(18, pretrained=True, base_width=64)

# 轻量版: ~2.8M 参数 (通道数减半)
model_lite = resnet(18, pretrained=True, base_width=32)

# 极简版: ~0.7M 参数 (通道数 1/4)
model_tiny = resnet(18, pretrained=True, base_width=16)
```

### 文件 2: `models/simopa.py`

从 `OPA/object_place_net.py` 复制，去掉 `from config import opt`。

```python
class SimOPA(nn.Module):
    def __init__(self, backbone='resnet18', base_width=64, num_classes=2,
                 pretrained=True, pretrained_weight=None):
        super().__init__()
        
        resnet_layers = int(backbone.split('resnet')[-1])
        full_backbone = resnet(
            resnet_layers,
            pretrained=pretrained,
            pretrained_weight=pretrained_weight,
            base_width=base_width,
            without_mask=False,
            in_channels=4
        )
        
        # 去掉 avgpool 和 fc
        features = list(full_backbone.children())[:-2]
        self.backbone = nn.Sequential(*features)
        
        # 分类头
        global_feature_dim = base_width * 8  # 512 for base_width=64, 256 for 32
        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            global_feature_dim = base_width * 8 * 4  # Bottleneck expansion
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(global_feature_dim, num_classes, bias=False)
    
    def forward(self, img_cat):
        # img_cat: [B, 4, 256, 256]
        feat = self.backbone(img_cat)      # [B, 8w, 8, 8]
        feat = self.pool(feat)             # [B, 8w, 1, 1]
        feat = feat.flatten(1)             # [B, 8w]
        logits = self.fc(feat)             # [B, 2]
        return logits
```

还要写一个推理包装类，对标原来 `simopa.py` 里的 `ObjectPlacementAssessmentModel`：
```python
class SimOPAScorer:
    """加载 SimOPA 权重，对外提供 score(image, mask) -> float 接口"""
    def __init__(self, weight_path, base_width=64, device='cuda'):
        self.device = torch.device(device)
        self.model = SimOPA(base_width=base_width, pretrained=False)
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        self.model = self.model.eval().to(self.device)
        self.image_size = 256
        
    @torch.no_grad()
    def score(self, composite_image, mask_image):
        """输入 PIL Image, 返回 float [0,1]"""
        # Resize + ToTensor + concat → [1, 4, 256, 256]
        img = self._preprocess(composite_image, mask_image)
        logits = self.model(img)
        return torch.softmax(logits, dim=-1)[0, 1].item()
```

**关键**：现在可以同时加载两个不同 `base_width` 的模型在内存中对比：
```python
scorer_full = SimOPAScorer('models/weights/simopa.pth', base_width=64)
scorer_lite = SimOPAScorer('models/weights/simopa.pth', base_width=32)
```

**但是注意**：原版 simopa.pth 的权重形状是 base_width=64 的。加载到 base_width=32 的模型时，只有形状匹配的层会被加载（layer1 部分匹配，更深层不匹配）。所以轻量模型需要**重新训练或用知识蒸馏**才能达到好的效果。这个我们后面专项讨论——先用原版模型作为"评分后端"，轻量模型作为"对比实验"。

---

## 第二步：候选位置生成 + 合成 + 评分流水线

### 文件 3: `pipeline/candidates.py`

```python
def generate_candidates(bg_width, bg_height, fg_width, fg_height,
                        grid_size=5, n_scales=3):
    """
    在背景上生成候选放置位置的网格。
    
    Args:
        bg_width, bg_height: 背景尺寸
        fg_width, fg_height: 前景原始尺寸
        grid_size: 网格密度 (5 → 5×5=25个候选)
        n_scales: 尺度数量 (3 → 0.8×, 1.0×, 1.2×)
    
    Returns:
        list of dict: [{'bbox': [x1,y1,x2,y2], 'scale': s}, ...]
    """
    candidates = []
    scales = [0.8, 1.0, 1.2][:n_scales]  # 可调
    
    for scale in scales:
        w = int(fg_width * scale)
        h = int(fg_height * scale)
        # 确保不超出背景
        if w >= bg_width or h >= bg_height:
            continue
        step_x = (bg_width - w) / (grid_size + 1)
        step_y = (bg_height - h) / (grid_size + 1)
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x1 = int(step_x * i)
                y1 = int(step_y * j)
                candidates.append({
                    'bbox': [x1, y1, x1 + w, y1 + h],
                    'scale': scale,
                })
    return candidates
```

### 文件 4: `pipeline/composite.py`

```python
def make_composite(bg_image, fg_image, fg_mask, bbox):
    """
    将前景合成到背景的指定 bbox 处。
    
    Args:
        bg_image:  PIL Image (背景)
        fg_image:  PIL Image (前景, RGB)
        fg_mask:   PIL Image (遮罩, L 模式, 255=前景 0=背景)
        bbox:      [x1, y1, x2, y2]
    
    Returns:
        composite: PIL Image (合成图)
        mask_full: PIL Image (全图尺寸的遮罩, bbox 外为 0)
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    fg_resized = fg_image.resize((w, h), Image.LANCZOS)
    mask_resized = fg_mask.resize((w, h), Image.LANCZOS)
    
    # 合成
    bg = bg_image.copy()
    fg_arr = np.array(fg_resized).astype(np.float32)
    bg_arr = np.array(bg).astype(np.float32)
    mask_arr = np.array(mask_resized).astype(np.float32) / 255.0  # [0,1]
    mask_3ch = np.stack([mask_arr] * 3, axis=-1)
    
    bg_arr[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (
        fg_arr * mask_3ch + bg_arr[bbox[1]:bbox[3], bbox[0]:bbox[2]] * (1 - mask_3ch)
    )
    
    # 生成全图 mask (用于 SimOPA 的 4ch 输入)
    mask_full = np.zeros((bg.height, bg.width), dtype=np.uint8)
    mask_full[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (
        np.array(mask_resized) > 127
    ).astype(np.uint8) * 255
    
    return Image.fromarray(bg_arr.astype(np.uint8)), Image.fromarray(mask_full)
```

### 文件 5: `pipeline/scorer.py`

```python
class PlacementScorer:
    """对一组候选位置进行评分并排序"""
    
    def __init__(self, model: SimOPAScorer):
        self.model = model
    
    def score_candidates(self, bg, fg, fg_mask, candidates):
        """
        Args:
            bg: PIL Image
            fg: PIL Image  
            fg_mask: PIL Image
            candidates: list of dict (from generate_candidates)
        
        Returns:
            list of dict: candidates + 'score' field, sorted by score descending
        """
        results = []
        for cand in candidates:
            composite, mask = make_composite(bg, fg, fg_mask, cand['bbox'])
            score = self.model.score(composite, mask)
            results.append({**cand, 'score': score, 'composite': composite})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
```

---

## 第三步：Gradio 应用

### 文件 6: `app.py`

```python
import gradio as gr
import numpy as np
from PIL import Image
import torch

from models.simopa import SimOPAScorer
from pipeline.candidates import generate_candidates
from pipeline.scorer import PlacementScorer

# -- 加载模型 (启动时加载一次) --
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scorer = SimOPAScorer('models/weights/simopa.pth', base_width=64, device=device)
pipeline = PlacementScorer(scorer)

def analyze_placement(bg_img, fg_img, grid_size, n_scales):
    """主分析函数：Gradio 的回调"""
    bg = Image.fromarray(bg_img).convert('RGB')
    fg = Image.fromarray(fg_img).convert('RGB')
    
    # 从 alpha 通道提取 mask
    if fg_img.shape[-1] == 4:
        fg_mask = Image.fromarray(fg_img[:, :, 3]).convert('L')
    else:
        # 无 alpha: 整个图都是前景
        fg_mask = Image.new('L', fg.size, 255)
    
    # 生成候选
    candidates = generate_candidates(bg.width, bg.height,
                                      fg.width, fg.height,
                                      grid_size=int(grid_size),
                                      n_scales=int(n_scales))
    
    # 评分
    results = pipeline.score_candidates(bg, fg, fg_mask, candidates)
    
    # 构建输出
    top3 = results[:3]
    gallery = [(r['composite'], f"#{i+1} score={r['score']:.3f}") 
               for i, r in enumerate(top3)]
    
    # 热力图: 在背景上叠加分数
    heatmap = bg.copy()
    # ... 画候选位置的分数标记
    
    return heatmap, gallery, format_results_table(results[:10])

# Gradio 界面
with gr.Blocks(title="物体放置助手") as demo:
    gr.Markdown("# 物体放置助手 — Object Placement Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            bg_input = gr.Image(label="背景图", type="numpy")
            fg_input = gr.Image(label="前景物体 (支持透明PNG)", type="numpy")
            grid_slider = gr.Slider(3, 9, value=5, step=1, label="搜索密度 (网格大小)")
            scale_slider = gr.Slider(1, 5, value=3, step=1, label="尺度数量")
            btn = gr.Button("🔍 分析最佳放置位置", variant="primary")
        
        with gr.Column(scale=2):
            heatmap_out = gr.Image(label="评分热力图")
            gallery_out = gr.Gallery(label="Top-3 推荐放置", columns=3)
            table_out = gr.Dataframe(label="Top-10 详细评分")
    
    btn.click(
        fn=analyze_placement,
        inputs=[bg_input, fg_input, grid_slider, scale_slider],
        outputs=[heatmap_out, gallery_out, table_out]
    )

demo.launch()
```

---

## 第四步：进阶项具体实现

### 4.1 模型解释 (Grad-CAM) — `grad_cam.py`

```python
class GradCAM:
    """对 SimOPA 的最后一个卷积层做 Grad-CAM"""
    
    def __init__(self, model: SimOPA):
        self.model = model
        self.activations = {}
        self.gradients = {}
        
        # 注册 hook 到 backbone 的最后一层 (layer4)
        target_layer = model.backbone[-1]  # layer4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations['target'] = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients['target'] = grad_output[0].detach()
    
    def generate(self, img_cat, class_idx=1):
        """
        Args:
            img_cat: [1, 4, 256, 256]
            class_idx: 0=unreasonable, 1=reasonable
        Returns:
            heatmap: [256, 256] numpy array
        """
        self.model.zero_grad()
        logits = self.model(img_cat)
        logits[0, class_idx].backward()
        
        # 权重 = 梯度全局平均
        grads = self.gradients['target']  # [1, C, 8, 8]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # CAM = ReLU(Σ w_k * A_k)
        acts = self.activations['target']  # [1, C, 8, 8]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, 8, 8]
        cam = torch.relu(cam)
        
        # 上采样到 256 × 256
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear')
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
```

**Gradio 中的展示**：在 `analyze_placement()` 返回结果时，额外对 Top-1 的合成图做 Grad-CAM，叠加显示为热力图。

### 4.2 多模型串联 (FOPA 先粗筛 → SimOPA 精排) — `pipeline/two_stage.py`

```
流程:
  背景 + 前景 + mask
       ↓
  FOPA (libcom FOPAHeatMapModel)  → 256×256 热力图 (每个像素的合理性)
       ↓
  取 Top-K 局部极大值区域 (如 K=10)
       ↓
  对每个区域生成合成图 → SimOPA 精排
       ↓
  Top-3 推荐 + 对比 FOPA only vs FOPA+SimOPA 排序差异
```

```python
from libcom import FOPAHeatMapModel

class TwoStageScorer:
    def __init__(self, fopa_model, simopa_scorer):
        self.fopa = fopa_model
        self.simopa = simopa_scorer
    
    def score(self, bg, fg, fg_mask):
        # Stage 1: FOPA 产生热力图
        box_list, heatmap_list = self.fopa(
            bg, fg, fg_mask, 
            composite_num=20  # 粗筛 20 个候选
        )
        
        # Stage 2: SimOPA 对每个候选精排
        results = []
        for bbox in box_list:
            # bbox 格式: [x, y, w, h] → [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            cand = {'bbox': [x1, y1, x1+w, y1+h]}
            composite, mask = make_composite(bg, fg, fg_mask, cand['bbox'])
            score = self.simopa.score(composite, mask)
            results.append({**cand, 'score': score, 'composite': composite})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
```

**对比展示**：在 Gradio 界面上显示两列——"FOPA 原始排序" vs "FOPA+SimOPA 精排"，让用户看到两阶段筛选的效果。

### 4.3 复杂交互 — 拖拽 + 实时评分

在 Gradio 中实现拖拽比较复杂，可以用 `gr.Image` 的 `tool="sketch"` 或改用 PySide6 做桌面应用。一个折中方案：

```python
# 用 gr.Image 作为背景，用户用鼠标点击某个位置
# 系统对该位置做即时评分

def on_click(evt: gr.SelectData, bg_img, fg_img):
    x, y = evt.index[0], evt.index[1]  # 点击坐标
    fg_w, fg_h = fg_img.width, fg_img.height
    bbox = [x, y, x + fg_w, y + fg_h]
    composite, mask = make_composite(bg_img, fg_img, fg_mask, bbox)
    score = scorer.score(composite, mask)
    # 返回带标注的合成图和分数
    return annotated_composite, f"{score:.3f}"
```

---

## 第五步：测试案例

### 文件 7: `test_cases.py`

```python
"""运行一组预定义的测试案例并保存结果"""

TEST_CASES = [
    # (名称, 背景路径, 前景路径, 描述, 预期)
    ("合理-草地上的狗", "assets/bg/grass.jpg", "assets/fg/dog.png",
     "狗放在草地上，尺度适中", "应得高分 >0.6"),
    ("合理-桌上的杯子", "assets/bg/desk.jpg", "assets/fg/cup.png",
     "杯子放在桌面区域", "应得高分 >0.6"),
    ("错误-天空中的狗", "assets/bg/sky.jpg", "assets/fg/dog.png",
     "狗浮在天空中", "应得低分 <0.3"),
    ("错误-超出边界", "assets/bg/room.jpg", "assets/fg/chair.png",
     "椅子一半在画面外", "应得低分 <0.3"),
    ("边界-部分遮挡", "assets/bg/street.jpg", "assets/fg/person.png",
     "行人被路灯杆部分遮挡", "中等分数 0.3-0.6"),
    ("尺度-过小的猫", "assets/bg/livingroom.jpg", "assets/fg/cat.png",
     "猫缩放到极小放在角落", "较低分 <0.4"),
    # ... 共 10-12 组
]

def run_tests(scorer_full, scorer_lite=None):
    """对每组案例，用原版和轻量模型分别评分，输出对比表"""
    results = []
    for name, bg_path, fg_path, desc, expected in TEST_CASES:
        # 生成 3 个候选位置 (预定义的: 好/一般/差)
        scores_full = []
        scores_lite = []
        for placement_type in ['good', 'mediocre', 'bad']:
            bbox = PREDEFINED_BBOXES[name][placement_type]
            composite, mask = make_composite(bg, fg, fg_mask, bbox)
            scores_full.append(scorer_full.score(composite, mask))
            if scorer_lite:
                scores_lite.append(scorer_lite.score(composite, mask))
        results.append({...})
    
    # 输出对比表: Spearman 相关系数, 推理时间
    return results
```

---

## 具体实施顺序（按天排）

```
第 1-2 天: 环境搭建
  - 新建 placement_app/ 目录结构
  - 从 OPA 复制 simopa.pth 到 models/weights/
  - 写 requirements.txt, 安装依赖
  - 验证: 能在本地加载 simopa.pth 跑通一张图的评分

第 3-4 天: 解耦模型
  - 写 models/resnet_4ch.py (加 base_width)
  - 写 models/simopa.py (SimOPA + SimOPAScorer)
  - 验证: base_width=64 模型评分结果与 OPA 原始 simopa.py 一致

第 5-6 天: 流水线
  - 写 pipeline/candidates.py (候选生成)
  - 写 pipeline/composite.py (合成)
  - 写 pipeline/scorer.py (评分+排序)
  - 验证: 对一组 bg/fg 跑出候选排序结果

第 7-9 天: Gradio 应用
  - 写 app.py 基础版 (上传→网格评分→展示 Top-3)
  - 加交互: 点击位置即时评分
  - 加可视化: 热力图叠加
  - 准备 5-10 组预设素材

第 10-12 天: 进阶项
  - 选做1: 模型解释 Grad-CAM (grad_cam.py)
  - 选做2: 两阶段 FOPA+SimOPA (pipeline/two_stage.py)
  - 选做3: 轻量模型对比 (base_width=32 模型)

第 13-14 天: 测试 + 报告
  - 写 test_cases.py, 跑 10+ 组案例
  - 截图、录屏
  - 写 README
  - 做 PPT

第 15 周: 课堂展示
  - 现场演示完整流程
  - 展示 Grad-CAM 解释图
  - 展示原版 vs 轻量版对比
```

---

## 关键风险和应对

| 风险 | 应对 |
|------|------|
| `base_width=32` 轻量模型无法直接加载 `simopa.pth` 权重 | 用 base_width=64 的 SimOPA 作为主力评分模型；另训练一个轻量版（用 OPA 数据集微调 5-10 epoch）做对比实验 |
| FOPA 模型依赖较多（mmdet, diffusers 等） | 若安装困难，改为只用 SimOPA。两阶段改为：网格粗筛（大步长5×5）→ 细筛（小步长在 top-3 周围局部搜索） |
| Gradio 拖拽交互不够精细 | 用 "点击位置 + 滑块微调坐标" 的组合替代 |
| 本地 GPU 显存不足 | SimOPA 仅 ~11M 参数，CPU 上推理也只需 ~50ms/图，完全可用 CPU 模式 |
