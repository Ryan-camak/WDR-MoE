# 🐧Please note that this file has been modified by Tencent on 2026/02/07. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Expert modules for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import FlopsUtils, get_safe_groups


# ==========================================
# Optimized expert modules
# ==========================================
class OptimizedSimpleExpert(nn.Module):
    """Use GroupNorm instead of BatchNorm to improve stability for small batches."""

    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.conv[3], (1, self.hidden_dim, H, W))
        return flops


class FusedGhostExpert(nn.Module):
    """Fused Ghost expert that reduces memory traffic by combining operations."""

    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Use GroupNorm to improve stability
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(min(num_groups, init_channels), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(min(num_groups, new_channels), new_channels),
            nn.SiLU(inplace=True)
        )
        self.init_channels = init_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.cheap_operation[0], (1, self.init_channels, H, W))
        return flops


class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x): return self.conv(x)

    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)


class SpatialExpert(nn.Module):
    """Expert network with 3x3 spatial convolution, enabling experts to learn spatial patterns."""
    def __init__(self, in_ch, out_ch, expand_ratio=2):
        super().__init__()
        hid = int(in_ch * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),  # DW spatial conv
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        return FlopsUtils.count_conv2d(self.conv, input_shape)


class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv, input_shape)
        # Compute input shape to cheap op (output of primary conv)
        p_out = self.primary_conv[0].out_channels
        flops += FlopsUtils.count_conv2d(self.cheap_operation, (B, p_out, H, W))
        return flops


class InvertedResidualExpert(nn.Module):
    """
    Highly efficient expert module: Uses Inverted Residual structure (MobileNetV2 style).
    2-3x faster than standard convolution experts, fewer parameters, stronger non-linearity.
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=3):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # 1. Pointwise Expand
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # 2. Depthwise Spatial
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, 
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # 3. Pointwise Project
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        return FlopsUtils.count_conv2d(self.conv, input_shape)


class WaveletFrequencyExpert(nn.Module):
    """Frequency-specialized expert using fixed Haar wavelet decomposition."""

    _VALID_BANDS = ("ll", "lh", "hl", "hh")

    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio=2,
        frequency_band="ll",
    ):
        super().__init__()
        band = str(frequency_band).lower()
        if band not in self._VALID_BANDS:
            raise ValueError(f"Unsupported frequency_band='{frequency_band}'. Expected one of {self._VALID_BANDS}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_band = band
        self.band_index = self._VALID_BANDS.index(band)

        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        low = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.float32)
        high = torch.tensor([sqrt2_inv, -sqrt2_inv], dtype=torch.float32)
        ll = torch.outer(low, low)
        lh = torch.outer(low, high)
        hl = torch.outer(high, low)
        hh = torch.outer(high, high)
        haar = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer("haar_filters", haar, persistent=False)

    def _haar_dwt(self, x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        filters = self.haar_filters.repeat(c, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=c)
        coeffs = coeffs.view(b, c, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, self.band_index, :, :]

    def forward(self, x):
        target_h, target_w = x.shape[-2:]
        x_band = self._haar_dwt(x)
        out = self.conv(x_band)
        if out.shape[-2:] != (target_h, target_w):
            out = F.interpolate(out, size=(target_h, target_w), mode="nearest")
        return out

    def compute_flops(self, input_shape):
        b, c, h, w = input_shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        return FlopsUtils.count_conv2d(self.conv, (b, c, h2, w2))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(EfficientExpertGroup, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if not hasattr(self, "conv"):
            out_c = x.shape[1]
            self.conv = DepthwiseSeparableConv(x.shape[1], out_c, 3, 1)
        return self.conv(x)

