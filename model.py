"""
CAN + CBAM — Context-Aware Network with Convolutional Block Attention Module
for unified crowd density estimation.

Architecture:
    VGG-16 Frontend → CBAM → Scale-Aware Module (multi-scale parallel dilated
    convolutions) → Context Aggregation → CBAM → Backend → CBAM → Density Map

Key difference from CSRNet:
    CAN uses PARALLEL multi-scale dilated convolutions (d=1,2,3,6) with
    learned context aggregation, rather than serial dilated convolutions.
    This captures both local and global context simultaneously.

References:
    - Liu et al., "Context-Aware Crowd Counting", CVPR 2019
    - Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────────────────────
# CBAM: Convolutional Block Attention Module
# ──────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Squeezes spatial dimensions via global average & max pooling, then
    processes through a shared MLP to produce per-channel attention weights.

    Input:  (B, C, H, W)
    Output: (B, C, 1, 1) — channel attention weights
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction_ratio, 1)

        # Shared MLP (implemented as 1×1 convolutions)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Compresses channel dimension via average & max pooling across channels,
    concatenates the two, and applies a convolution to produce a spatial
    attention map.

    Input:  (B, C, H, W)
    Output: (B, 1, H, W) — spatial attention map
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies channel attention then spatial attention:
        x' = Channel_Attention(x) ⊗ x
        x'' = Spatial_Attention(x') ⊗ x'

    Args:
        in_channels:     Number of input channels
        reduction_ratio: Reduction ratio for channel attention MLP
        kernel_size:     Kernel size for spatial attention conv
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Scale-Aware Module (SAM) — Core CAN component
# ──────────────────────────────────────────────────────────────────────────────

class ScaleAwareModule(nn.Module):
    """
    Multi-scale feature extraction using parallel dilated convolutions.

    This is the key differentiator of CAN. Instead of serial dilated convs
    (like CSRNet), we use PARALLEL branches with different dilation rates
    to capture context at multiple scales simultaneously.

    Architecture:
        Input (512ch) → 4 parallel branches:
            Branch 1: 3×3 conv, dilation=1  (local details)
            Branch 2: 3×3 conv, dilation=2  (medium context)
            Branch 3: 3×3 conv, dilation=3  (large context)
            Branch 4: 3×3 conv, dilation=6  (global context)
        → Concatenate → 1×1 Conv (fuse) → Output (512ch)

    Args:
        in_channels:  Number of input channels (default: 512)
        out_channels: Number of output channels (default: 512)
        dilations:    Tuple of dilation rates for parallel branches
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 512,
                 dilations: tuple = (1, 2, 4, 12)):
        super().__init__()

        branch_channels = in_channels // len(dilations)

        self.branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                # Added an extra layer per branch for better feature extraction
                nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # Context aggregation: fuse multi-scale features
        total_branch_channels = branch_channels * len(dilations)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_branch_channels, out_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Residual connection if dimensions match
        self.use_res = in_channels == out_channels

        # Learnable scale weights for weighted aggregation
        self.scale_weights = nn.Parameter(
            torch.ones(len(dilations)) / len(dilations)
        )

    def forward(self, x):
        identity = x

        # Extract features at multiple scales in parallel
        branch_outputs = []
        weights = F.softmax(self.scale_weights, dim=0)

        for i, branch in enumerate(self.branches):
            branch_out = branch(x) * weights[i]
            branch_outputs.append(branch_out)

        # Concatenate and fuse
        multi_scale = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(multi_scale)

        if self.use_res:
            fused = fused + identity

        return fused


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary Density Classifier
# ──────────────────────────────────────────────────────────────────────────────

class DensityClassifier(nn.Module):
    """
    Auxiliary binary classifier: predicts whether a scene is dense or sparse.

    Uses global average pooling on features to classify the density level.
    This auxiliary loss helps the model learn better representations that
    distinguish between crowded and sparse scenes.

    During training, the classification loss is added to the density MSE loss.
    During inference, the classification output is available for analysis.

    Args:
        in_channels: Number of input feature channels
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """Returns logit (single value, use sigmoid for probability)."""
        pooled = self.pool(x).flatten(1)  # (B, C)
        logit = self.classifier(pooled)   # (B, 1)
        return logit


# ──────────────────────────────────────────────────────────────────────────────
# CAN + CBAM: Context-Aware Network
# ──────────────────────────────────────────────────────────────────────────────

class CANCBAM(nn.Module):
    """
    Context-Aware Network (CAN) enhanced with CBAM attention for unified
    crowd density estimation across both dense and sparse scenes.

    Architecture:
        ┌─────────────────────────────────────────────────────────────────┐
        │  VGG-16 Frontend (conv1_1 → conv4_3)   [pretrained, 23 layers] │
        ├─────────────────────────────────────────────────────────────────┤
        │  ★ CBAM (512 channels)  ← suppresses background features      │
        ├─────────────────────────────────────────────────────────────────┤
        │  Scale-Aware Module: 4 parallel dilated conv branches          │
        │    d=1 (local) | d=2 (medium) | d=3 (large) | d=6 (global)    │
        │    → Weighted concatenation → 1×1 fusion → 512ch              │
        ├─────────────────────────────────────────────────────────────────┤
        │  ★ CBAM (512 channels)  ← post-context attention              │
        ├─────────────────────────────────────────────────────────────────┤
        │  [Auxiliary branch] → DensityClassifier (dense/sparse)         │
        ├─────────────────────────────────────────────────────────────────┤
        │  Backend: 512→256→128→64 (dilated convolutions)                │
        ├─────────────────────────────────────────────────────────────────┤
        │  ★ CBAM (64 channels)   ← final feature refinement            │
        ├─────────────────────────────────────────────────────────────────┤
        │  1×1 Conv → Density Map (1 channel)                            │
        └─────────────────────────────────────────────────────────────────┘

    Args:
        pretrained:       Use ImageNet-pretrained VGG-16 frontend
        reduction_ratio:  CBAM channel attention reduction ratio
        cbam_kernel:      CBAM spatial attention kernel size
        dilations:        Dilation rates for Scale-Aware Module
        use_aux_classifier: Enable auxiliary dense/sparse classifier
    """

    def __init__(self, pretrained: bool = True, reduction_ratio: int = 16,
                 cbam_kernel: int = 7, dilations: tuple = (1, 2, 3, 6),
                 use_aux_classifier: bool = True):
        super().__init__()
        self.use_aux_classifier = use_aux_classifier

        # ── Frontend: VGG-16 feature extractor (layers 0-22) ──
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[:23])

        # Fine-tune frontend (lower LR recommended via param groups)
        for param in self.frontend.parameters():
            param.requires_grad = True

        # ★ CBAM after frontend (512 channels from conv4_3)
        self.cbam_frontend = CBAM(512, reduction_ratio, cbam_kernel)

        # ── Scale-Aware Module: multi-scale parallel dilated convolutions ──
        self.scale_aware = ScaleAwareModule(
            in_channels=512, out_channels=512, dilations=dilations
        )

        # ★ CBAM after context aggregation (512 channels)
        self.cbam_context = CBAM(512, reduction_ratio, cbam_kernel)

        # ── Auxiliary Density Classifier (dense vs sparse) ──
        if self.use_aux_classifier:
            self.density_classifier = DensityClassifier(in_channels=512)

        # ── Backend: Progressive channel reduction with dilation ──
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ★ CBAM after backend (64 channels)
        self.cbam_backend = CBAM(64, reduction_ratio, cbam_kernel)

        # ── Output: 1×1 conv to produce single-channel density map ──
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialize non-pretrained weights
        self._initialize_weights(self.scale_aware)
        self._initialize_weights(self.backend)
        self._initialize_weights(self.output_layer)
        if self.use_aux_classifier:
            self._initialize_weights(self.density_classifier)

    def forward(self, x):
        """
        Forward pass.

        Input:  x — (B, 3, H, W) image tensor
        Output: density_map — (B, 1, H/8, W/8) density map
                aux_logit   — (B, 1) dense/sparse classification logit
                              (only if use_aux_classifier=True)

        The multi-scale context features are key to handling both dense
        and sparse crowds with a single model.
        """
        # VGG-16 frontend
        x = self.frontend(x)

        # ★ CBAM 1: suppress background after VGG feature extraction
        x = self.cbam_frontend(x)

        # Scale-Aware Module: multi-scale parallel feature extraction
        x = self.scale_aware(x)

        # ★ CBAM 2: post-context attention refinement
        x = self.cbam_context(x)

        # Auxiliary density classifier (branch off here before backend)
        aux_logit = None
        if self.use_aux_classifier:
            aux_logit = self.density_classifier(x)

        # Backend: progressive channel reduction
        x = self.backend(x)

        # ★ CBAM 3: final feature refinement
        x = self.cbam_backend(x)

        # 1×1 output convolution → density map
        density_map = self.output_layer(x)

        if self.use_aux_classifier:
            return density_map, aux_logit
        return density_map

    @staticmethod
    def _initialize_weights(module):
        """Kaiming initialization for conv layers, Xavier for linear."""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_count(self, density_map):
        """Sum density map to get estimated count."""
        return density_map.sum(dim=[1, 2, 3])

    def count_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable

    def get_param_groups(self, frontend_lr_scale=0.1):
        """
        Return parameter groups with different learning rates.
        Frontend gets lower LR since it's pretrained.

        Args:
            frontend_lr_scale: Multiply base LR by this for frontend params

        Returns:
            list of param group dicts for optimizer
        """
        frontend_params = list(self.frontend.parameters())
        other_params = [p for name, p in self.named_parameters()
                        if not name.startswith('frontend')]

        return [
            {'params': frontend_params, 'lr_scale': frontend_lr_scale},
            {'params': other_params, 'lr_scale': 1.0},
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Model summary utility
# ──────────────────────────────────────────────────────────────────────────────

def print_model_summary(model: CANCBAM):
    """Print a detailed summary of the CAN+CBAM model."""
    total, trainable = model.count_parameters()
    print("=" * 65)
    print("  CAN + CBAM — Model Summary")
    print("=" * 65)
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frontend layers:      {len(list(model.frontend.children())):>12}")
    print(f"  SAM branches:         {len(model.scale_aware.branches):>12}")
    print(f"  Backend layers:       {len(list(model.backend.children())):>12}")
    print(f"  CBAM modules:         {'3':>12} (frontend, context, backend)")
    print(f"  Aux classifier:       {str(model.use_aux_classifier):>12}")
    print("=" * 65)

    # Test forward pass
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        result = model(dummy)
        if isinstance(result, tuple):
            out, aux = result
            print(f"  Input shape:          (1, 3, 256, 256)")
            print(f"  Output shape:         {tuple(out.shape)}")
            print(f"  Aux classifier shape: {tuple(aux.shape)}")
        else:
            out = result
            print(f"  Input shape:          (1, 3, 256, 256)")
            print(f"  Output shape:         {tuple(out.shape)}")
    print(f"  Output spatial:       {out.shape[2]}×{out.shape[3]} "
          f"(downsampled 8×)")

    # Print scale-aware weights
    weights = F.softmax(model.scale_aware.scale_weights, dim=0)
    weight_str = ", ".join(f"{w:.3f}" for w in weights.detach().cpu().numpy())
    print(f"  SAM scale weights:    [{weight_str}]")
    print("=" * 65)


if __name__ == "__main__":
    print("Testing CAN + CBAM architecture...\n")

    # Test with aux classifier
    net = CANCBAM(pretrained=False, use_aux_classifier=True)
    print_model_summary(net)

    # Test without aux classifier
    print("\n--- Without Auxiliary Classifier ---")
    net2 = CANCBAM(pretrained=False, use_aux_classifier=False)
    print_model_summary(net2)
