"""
TopNet: Transformer-based Object Placement Network (CVPR 2023).

Produces a per-pixel rationality heatmap from a single forward pass
through a dual-encoder + Transformer + UNet-decoder architecture.

Checkpoint: ``best_weight.pth`` (self-contained, ~430 MB, 336 keys).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.ndimage import maximum_filter


# -------------------------------------------------------------------
#  BasicConv2d  (Conv + BN + ReLU, wrapped in Sequential named "basicconv")
# -------------------------------------------------------------------
class BasicConv2d(nn.Module):
    """Must match TopNet's checkpoint key structure exactly."""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


# -------------------------------------------------------------------
#  ResNet backbone (minimal version matching TopNet checkpoint)
# -------------------------------------------------------------------
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self._init_weights(zero_init_residual)
        if zero_init_residual:
            self._zero_last_bn(block)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _zero_last_bn(self, block):
        for m in self.modules():
            if isinstance(m, _BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# -------------------------------------------------------------------
#  Build encoder sub-modules exactly as TopNet does
# -------------------------------------------------------------------
def _build_bg_encoder():
    """bg_encoder1..5 — from pretrained 4ch ResNet18."""
    model = _ResNet(_BasicBlock, [2, 2, 2, 2])
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    # Weights will be loaded from the TopNet checkpoint
    div_2 = nn.Sequential(*list(model.children())[:3])     # conv1, bn1, relu
    div_4 = nn.Sequential(*list(model.children())[3:5])    # maxpool, layer1
    div_8 = model.layer2
    div_16 = model.layer3
    div_32 = model.layer4
    return div_2, div_4, div_8, div_16, div_32


def _build_fg_encoder():
    """fg_encoder1..6 — from 4ch ResNet18 with greyscale-init 4th channel."""
    model = _ResNet(_BasicBlock, [2, 2, 2, 2])
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    div_1 = nn.Sequential(*list(model.children())[:1])     # conv1
    div_2 = nn.Sequential(*list(model.children())[1:3])    # bn1, relu
    div_4 = nn.Sequential(*list(model.children())[3:5])    # maxpool, layer1
    div_8 = model.layer2
    div_16 = model.layer3
    div_32 = model.layer4
    return div_1, div_2, div_4, div_8, div_16, div_32


# -------------------------------------------------------------------
#  Transformer  (must match checkpoint key naming)
# -------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.flatten(x)
        x = self.model(x)
        return x.reshape(B, C, H, W)


class _TransformerModule(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads):
        super().__init__()
        self.MultiheadAttention = nn.MultiheadAttention(embedding_size, num_heads, batch_first=False)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = _MLP(1024 * 8 * 8, 128, 1024 * 8 * 8)

    def forward(self, x):
        x1 = self.layer_norm1(x)
        B, E, H, W = x1.shape
        x1 = x1.reshape(B, E, H * W).permute(0, 2, 1)
        attn_out, _ = self.MultiheadAttention(x1, x1, x1)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, E, H, W)
        x = x + attn_out
        x2 = self.layer_norm2(x)
        x = x + self.mlp(x2)
        return x


class _Transformer(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerModule(embedding_size, hidden_size, num_heads)
            for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------------------------------------------------
#  ObPlaNet_resnet18  (full model, checkpoint-compatible)
# -------------------------------------------------------------------
class ObPlaNet_resnet18(nn.Module):
    """TopNet's discriminator model for object placement."""
    def __init__(self):
        super().__init__()
        (self.bg_encoder1, self.bg_encoder2, self.bg_encoder4,
         self.bg_encoder8, self.bg_encoder16) = _build_bg_encoder()
        (self.fg_encoder1, self.fg_encoder2, self.fg_encoder4,
         self.fg_encoder8, self.fg_encoder16,
         self.fg_encoder32) = _build_fg_encoder()

        self.n_layers = 4
        self.Transformer = _Transformer(
            embedding_size=1024, hidden_size=8, num_heads=8, n_layers=self.n_layers)

        self.upconv32 = BasicConv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv8  = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4  = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2  = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1  = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv = nn.ConvTranspose2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(512, 2, 1)

    def forward(self, bg, fg, mask):
        """bg/fg: [B,3,256,256], mask: [B,1,256,256] → [B,2,256,256]"""
        black = torch.zeros_like(mask)
        bg_4ch = torch.cat([bg, black], dim=1)
        fg_4ch = torch.cat([fg, mask], dim=1)

        bg1 = self.bg_encoder1(bg_4ch) # stride=2
        fg1 = self.fg_encoder1(fg_4ch)
        bg2 = self.bg_encoder2(bg1)
        fg2 = self.fg_encoder2(fg1)
        bg4 = self.bg_encoder4(bg2)
        fg4 = self.fg_encoder4(fg2)
        bg8 = self.bg_encoder8(bg4)
        fg8 = self.fg_encoder8(fg4)
        bg16 = self.bg_encoder16(bg8)
        fg16 = self.fg_encoder16(fg8)
        fg32 = self.fg_encoder32(fg16)

        x = torch.cat([bg16, fg32], dim=1)          # [B, 1024, 8, 8]
        x = self.Transformer(x)
        x = self.upconv32(x)
        x = _upsample_add(self.upconv16(x), bg8)
        x = _upsample_add(self.upconv8(x), bg4)
        x = _upsample_add(self.upconv4(x), bg2)
        x = _upsample_add(self.upconv2(x), bg1)
        x = self.upconv1(F.interpolate(x, scale_factor=2, mode='bilinear',
                                       align_corners=True))
        x = self.deconv(x)
        return self.classifier(x)


def _upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.shape[2:], mode='bilinear',
                              align_corners=False)
    return y


# -------------------------------------------------------------------
#  TopNetScorer — inference wrapper
# -------------------------------------------------------------------
class TopNetScorer:
    """Loads the TopNet checkpoint and provides heatmap + top-K extraction.

    Args:
        weight_path:  Path to ``best_weight.pth``.
        device:       'cuda' or 'cpu'.
    """

    def __init__(self, weight_path: str, device: str = 'cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.image_size = 256

        self.model = ObPlaNet_resnet18()
        state = torch.load(weight_path, map_location='cpu')

        # The checkpoint may have 'module.' prefix (DataParallel)
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', ''): v for k, v in state.items()}

        missing, unexpected = self.model.load_state_dict(state, strict=True)
        if missing:
            print(f'[TopNet] Missing keys: {len(missing)}')
        if unexpected:
            print(f'[TopNet] Unexpected keys: {len(unexpected)}')

        self.model = self.model.eval().to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'[TopNet] Loaded {weight_path}')
        print(f'         params={n_params:,}, device={self.device}')

        self._transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def heatmap(self, bg: Image.Image, fg: Image.Image,
                mask: Image.Image) -> np.ndarray:
        """
        Generate a placement rationality heatmap in one forward pass.

        Args:
            bg:   Background PIL image (RGB).
            fg:   Foreground PIL image (RGB).
            mask: Foreground mask (L), 255 = foreground.

        Returns:
            numpy array [256, 256] — values in [0, 1], higher = better.
        """
        bg_t = self._transform(bg.convert('RGB')).unsqueeze(0).to(self.device)
        fg_t = self._transform(fg.convert('RGB')).unsqueeze(0).to(self.device)
        mk_t = self._mask_transform(mask.convert('L')).unsqueeze(0).to(self.device)

        logits = self.model(bg_t, fg_t, mk_t)            # [1, 2, 256, 256]
        probs = F.softmax(logits, dim=1)                  # [1, 2, 256, 256]
        hmap = probs[0, 1].cpu().numpy()                  # [256, 256] reasonable class
        return hmap

    @torch.no_grad()
    def top_k_boxes(self, bg: Image.Image, fg: Image.Image,
                    mask: Image.Image, k: int = 15,
                    fg_w: int = None, fg_h: int = None) -> list:
        """
        Return the top-K placement bounding boxes from the heatmap.

        Uses local-maximum detection + greedy NMS with a small exclusion
        radius to ensure spatially diverse candidates.

        Returns:
            List of dicts: [{'bbox': [x1,y1,x2,y2], 'score': float}, ...]
        """
        hmap = self.heatmap(bg, fg, mask)
        h, w = hmap.shape  # 256, 256

        # Foreground bounding-box size in heatmap coordinates
        fw = int(fg.width / bg.width * w) if fg_w is None else fg_w
        fh = int(fg.height / bg.height * h) if fg_h is None else fg_h

        # Find pixels whose score is a local maximum within a small window
        nms_win = max(min(fw, fh) // 4, 3)
        local_max = (hmap == maximum_filter(hmap, size=nms_win))
        ys, xs = np.where(local_max)
        scores = hmap[ys, xs]

        # Sort by score descending
        order = np.argsort(scores)[::-1]
        boxes = []
        seen = set()  # (grid cell) to ensure spatial diversity

        for idx in order:
            cx, cy = int(xs[idx]), int(ys[idx])
            score = float(scores[idx])

            # Simple spatial dedup: at most one candidate per grid cell
            cell = (cx // max(fw // 2, 5), cy // max(fh // 2, 5))
            if cell in seen:
                continue
            seen.add(cell)

            # Map to original background coordinates
            x1 = int(cx / w * bg.width) - fg.width // 2
            y1 = int(cy / h * bg.height) - fg.height // 2
            x1 = max(0, min(x1, bg.width - fg.width))
            y1 = max(0, min(y1, bg.height - fg.height))
            boxes.append({
                'bbox': [x1, y1, x1 + fg.width, y1 + fg.height],
                'score': score,
            })
            if len(boxes) >= k:
                break

        return boxes
