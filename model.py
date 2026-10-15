"""Binary vegetation segmentation for 256 x 256 RGB inputs in [0, 1].

VegetationNet returns one logit per pixel. Apply sigmoid for probabilities.
The grouped sampling grid in model.py includes portions adapted from DySample:
https://github.com/tiny-smart/dysample
The structure-tensor direction and coherence modulation are local additions.

Upstream notice:

MIT License

Copyright (c) 2023 Wenze Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ChromaticBranch(nn.Module):

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        in_dim = 5
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-06
        r, g, b = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
        total = r + g + b + eps
        cr = r / total
        cg = g / total
        cb = b / total
        exg = 2.0 * cg - cr - cb
        ndi = (g - r) / (g + r + eps)
        features = torch.cat([cr, cg, cb, exg, ndi], dim=1)
        return self.conv(features)

class PhotometricBranch(nn.Module):

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        in_dim = 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        mean_y = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
        mean_y2 = F.avg_pool2d(y * y, kernel_size=5, stride=1, padding=2)
        var_y = F.relu(mean_y2 - mean_y * mean_y)
        std_y = torch.sqrt(var_y + 1e-06)
        feat = torch.cat([y, std_y], dim=1)
        return self.conv(feat)

class AppearanceEncoder(nn.Module):

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        branch_dim = out_channels // 2
        self.chroma_branch = ChromaticBranch(out_channels=branch_dim)
        self.photo_branch = PhotometricBranch(out_channels=branch_dim)
        self.gate = nn.Sequential(nn.Conv2d(branch_dim * 2, branch_dim * 2, kernel_size=1), nn.Sigmoid())
        self.out_conv = nn.Sequential(
            nn.Conv2d(branch_dim * 2 + 3, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_feat = self.chroma_branch(x)
        p_feat = self.photo_branch(x)
        combined = torch.cat([c_feat, p_feat], dim=1)
        gated = combined * self.gate(combined)
        return self.out_conv(torch.cat([x, gated], dim=1))

def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return (ll, lh, hl, hh)

def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    b, c, h, w = ll.shape
    out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out

class WaveletRefinement(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        reduced = max(8, channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(min(4, channels), channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(4, channels), channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        ll, lh, hl, hh = haar_dwt2d(x)
        eps = 1e-05
        directional_ratio = torch.abs(lh - hl) / (torch.abs(lh) + torch.abs(hl) + eps)
        c_gate = self.channel_gate(ll)
        s_gate = self.spatial_dw(directional_ratio)
        gate = c_gate * s_gate
        lh_clean = lh * gate
        hl_clean = hl * gate
        hh_clean = hh * gate
        reconstructed = haar_idwt2d(ll, lh_clean, hl_clean, hh_clean)
        out = self.refine(reconstructed)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return x[:, :, :h, :w] + out

class ConnectivityGeodesicDiffusion(nn.Module):

    def __init__(self, steps: int = 2) -> None:
        super().__init__()
        self.steps = steps

    def forward(self, core_prob: torch.Tensor, fringe_mask: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        prob = core_prob
        for _ in range(self.steps):
            neighbor_max = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
            gated = neighbor_max * fringe_mask * affinity
            prob = torch.maximum(prob, gated)
        return prob

class ConnectivityHead(nn.Module):

    def __init__(self, in_channels: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        self.diffusion_steps = diffusion_steps
        mid = max(16, in_channels // 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.GroupNorm(min(4, mid), mid),
            nn.GELU(),
        )
        self.core_head = nn.Conv2d(mid, 1, kernel_size=1)
        self.semantic_head = nn.Conv2d(mid, 1, kernel_size=1)
        self.affinity_conv = nn.Sequential(
            nn.Conv2d(mid + 3, 8, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.diffuser = ConnectivityGeodesicDiffusion(steps=diffusion_steps)
        self.fringe_weight = nn.Parameter(torch.tensor(1.0))

    def detect_chromatic_fringe(self, rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = (rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3])
        tot = (r + g + b).clamp(min=0.001)
        cr, cg, cb = (r / tot, g / tot, b / tot)
        dev_b = F.relu(cb - cg - 0.02)
        dev_m = F.relu((cr + cb) / 2.0 - cg - 0.03)
        soil_penalty = F.relu(cr - cg) * F.relu(cg - cb) * 5.0
        fringe = (dev_b * 3.5 + dev_m * 2.5 - soil_penalty).clamp(0.0, 1.0)
        return fringe

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.trunk(feat)
        core_logit = self.core_head(shared)
        core_prob = torch.sigmoid(core_logit)
        base_logit = self.semantic_head(shared)
        fringe_mask = self.detect_chromatic_fringe(rgb)
        affinity = self.affinity_conv(torch.cat([shared, rgb], dim=1))
        connected_prob = self.diffuser(core_prob, fringe_mask, affinity)
        fringe_annexed = connected_prob * fringe_mask
        final_logit = base_logit + self.fringe_weight * fringe_annexed
        return (final_logit, core_prob, fringe_annexed)

class StructureTensorOrientation(nn.Module):

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.to_intensity = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = self.to_intensity(x)
        working = gray if gray.dtype == torch.float64 else gray.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            padded = F.pad(working, (1, 1, 1, 1), mode='replicate')
            gx = F.conv2d(padded, self.sobel_x.to(working.dtype))
            gy = F.conv2d(padded, self.sobel_y.to(working.dtype))
            tensor = torch.cat((gx.square(), gy.square(), gx * gy), dim=1)
            tensor = F.avg_pool2d(F.pad(tensor, (1, 1, 1, 1), mode='replicate'), 3, 1)
            j_xx, j_yy, j_xy = tensor.chunk(3, dim=1)
            diff = j_xx - j_yy
            trace = j_xx + j_yy
            delta = torch.linalg.vector_norm(torch.cat((diff, 2 * j_xy), dim=1), dim=1, keepdim=True)
            eps = 1e-07
            coherence = (delta / (trace + eps)).clamp(0, 1)
            coherence = torch.where(trace > eps, coherence, torch.zeros_like(coherence))
            first = torch.cat((-j_xy, 0.5 * (diff + delta)), dim=1)
            second = torch.cat((0.5 * (delta - diff), -j_xy), dim=1)
            choose_first = first.square().sum(1, keepdim=True) >= second.square().sum(1, keepdim=True)
            vector = torch.where(choose_first, first, second)
            length = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
            fallback = torch.cat((torch.ones_like(trace), torch.zeros_like(trace)), dim=1)
            tangent = torch.where(length > eps, vector / length.clamp_min(eps), fallback)
        return (tangent.to(gray.dtype), coherence.to(gray.dtype))

class DirectionalUpsampler(nn.Module):

    def __init__(self, in_channels: int, scale: int = 2, groups: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.in_channels = in_channels
        out_channels = 2 * groups * scale ** 2
        self.offset_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.offset_conv.weight, std=0.001)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.structure_tensor = StructureTensorOrientation(in_channels)
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        mesh = torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2)
        return mesh.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        raw_offset = self.offset_conv(x)
        tangent, coherence = self.structure_tensor(x)
        offset = raw_offset.view(b, 2, self.groups * self.scale ** 2, h, w)
        tangent = tangent.unsqueeze(2)
        alpha = 0.75 * coherence.unsqueeze(2)
        projected = (offset * tangent).sum(1, keepdim=True) * tangent
        guided = ((1 - alpha) * offset + alpha * projected).reshape(b, -1, h, w)
        guided = guided + self.init_pos.to(dtype=guided.dtype)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing='ij')).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = x.new_tensor((w, h)).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + guided.view(b, 2, -1, h, w)) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(x.reshape(b * self.groups, -1, h, w), coords, mode='bilinear', align_corners=False, padding_mode='border')
        return sampled.view(b, c, self.scale * h, self.scale * w)

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)

class VegetationNet(nn.Module):

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.cpe = AppearanceEncoder(out_channels=c)
        self.enc0 = ConvBlock(c, c)
        self.enc1 = ConvBlock(c, c * 2)
        self.enc2 = ConvBlock(c * 2, c * 4)
        self.enc3 = ConvBlock(c * 4, c * 8)
        self.wcm0 = WaveletRefinement(c)
        self.wcm1 = WaveletRefinement(c * 2)
        self.wcm2 = WaveletRefinement(c * 4)
        self.up2 = DirectionalUpsampler(c * 8, scale=2)
        self.up1 = DirectionalUpsampler(c * 4, scale=2)
        self.up0 = DirectionalUpsampler(c * 2, scale=2)
        self.dec2 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec1 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec0 = ConvBlock(c * 2 + c, c)
        self.head = ConnectivityHead(in_channels=c)

    def forward(self, rgb):
        e0 = self.enc0(self.cpe(rgb))
        e1 = self.enc1(F.max_pool2d(e0, 2, 2))
        e2 = self.enc2(F.max_pool2d(e1, 2, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2, 2))
        s2, s1, s0 = (self.wcm2(e2), self.wcm1(e1), self.wcm0(e0))
        d2 = self.dec2(torch.cat([self.up2(e3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), s0], dim=1))
        return self.head(d0, rgb)[0]
