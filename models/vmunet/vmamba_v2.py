"""
vmamba_v2.py  —  VM-UNetV2 backbone
Upgrades over V1:
  • CBAM  : Convolutional Block Attention Module (channel + spatial attention)
  • SDI   : Semantics and Detail Infusion skip-connection module
  • Deep Supervision : last 3 decoder stages each produce an auxiliary logit map

All original Mamba primitives (SS2D, VSSBlock, VSSLayer, VSSLayer_up,
PatchEmbed2D, PatchMerging2D, PatchExpand2D, Final_PatchExpand2D) are kept
100 % intact.
"""

import time
import math
from functools import partial
from typing import Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# Helpers
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False,
                              with_Group=True, with_complex=False):
    import numpy as np

    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex
    flops = 0
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# Patch utilities  (unchanged from V1)
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========",
                  flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


# Mamba core  (SS2D, VSSBlock, VSSLayer, VSSLayer_up)
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(self.expand * self.d_model)
        self.dt_rank  = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj  = nn.Linear(self.d_model, self.d_inner * 2,
                                  bias=bias, **factory_kwargs)
        self.conv2d   = nn.Conv2d(in_channels=self.d_inner,
                                  out_channels=self.d_inner,
                                  groups=self.d_inner, bias=conv_bias,
                                  kernel_size=d_conv,
                                  padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner,
                                      copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj  = nn.Linear(self.d_inner, self.d_model,
                                   bias=bias, **factory_kwargs)
        self.dropout   = nn.Dropout(dropout) if dropout > 0. else None

    # ── static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32,
                                device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # ── forward passes ────────────────────────────────────────────────────────
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                              xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl,
                                   [self.dt_rank, self.d_state, self.d_state],
                                   dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight)
        xs  = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs  = Bs.float().view(B, K, -1, L)
        Cs  = Cs.float().view(B, K, -1, L)
        Ds  = self.Ds.float().view(-1)
        As  = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y   = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y    = torch.transpose(out_y[:, 1].view(B, -1, W, H),
                                  dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H),
                                  dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                              xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl,
                                   [self.dt_rank, self.d_state, self.d_state],
                                   dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight)
        xs  = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs  = Bs.float().view(B, K, -1, L)
        Cs  = Cs.float().view(B, K, -1, L)
        Ds  = self.Ds.float().view(-1)
        As  = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias, delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y   = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y    = torch.transpose(out_y[:, 1].view(B, -1, W, H),
                                  dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H),
                                  dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(
                     nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim,
                                   dropout=attn_drop_rate,
                                   d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        return input + self.drop_path(self.self_attention(self.ln_1(input)))


class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=drop_path[i] if isinstance(drop_path, list)
                     else drop_path,
                     norm_layer=norm_layer, attn_drop_rate=attn_drop,
                     d_state=d_state)
            for i in range(depth)])
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) \
            if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=drop_path[i] if isinstance(drop_path, list)
                     else drop_path,
                     norm_layer=norm_layer, attn_drop_rate=attn_drop,
                     d_state=d_state)
            for i in range(depth)])
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) \
            if upsample is not None else None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x


#  NEW V2 modules
# 1.  CBAM  — Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.

    Given feature map F ∈ R^{B×C×H×W}, we compute:
        M_c = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )

    and return M_c ⊗ F.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        B, C, H, W = x.shape
        avg = x.mean(dim=[2, 3])            # (B, C)
        mx  = x.amax(dim=[2, 3])            # (B, C)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention branch of CBAM.

    Concatenate channel-wise avg-pool and max-pool, then convolve
    to produce a single-channel mask M_s ∈ R^{B×1×H×W}.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    Full CBAM block: channel attention → spatial attention.

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# 2.  SDI  — Semantics and Detail Infusion module
class SDI(nn.Module):
    """
    SDI fuses all 4 encoder feature maps so that each decoder stage
    receives a rich skip signal combining both fine details (shallow maps)
    and high-level semantics (deep maps).

    Construction (per instance = one decoder stage):
        • One CBAM per encoder level  →  attention-weighted features
        • One 1×1 conv per level      →  channel alignment to `target_channels`
        • Spatial adaptation to match the target decoder resolution:
              shallower (larger) features → AdaptiveAvgPool2d (downsampling)
              same-resolution features   → Identity
              deeper (smaller) features  → bilinear upsample
        • One 3×3 "smooth" conv per level
        • Fuse via element-wise product (Hadamard) of all four aligned maps

    Args:
        encoder_channels : list of C_i  for the 4 encoder outputs
                           (ordered shallow→deep, i.e. dims=[96,192,384,768])
        target_channels  : output channel dimension for this SDI instance
                           (equals dims_decoder[inx] of the current stage)
        target_index     : which encoder level the current decoder stage
                           corresponds to spatially (0 = shallowest /
                           largest, 3 = deepest / smallest).
                           Used only to determine relative up/down-sampling.

    Forward:
        features : list of 4 tensors in NCHW format,
                   shapes (B, C_i, H_i, W_i)  — they arrive permuted from
                   the (B,H,W,C) Mamba convention inside VSSM.forward.
        Returns  : fused tensor of shape (B, target_channels, H_t, W_t)
                   in NCHW, which the caller should permute back to NHWC.
    """
    def __init__(self, encoder_channels: List[int],
                 target_channels: int,
                 target_index: int):
        super().__init__()
        assert len(encoder_channels) == 4, "Expect exactly 4 encoder stages"
        self.target_index = target_index

        # Per-level components
        self.cbam_layers   = nn.ModuleList()
        self.align_convs   = nn.ModuleList()
        self.smooth_convs  = nn.ModuleList()

        for in_c in encoder_channels:
            # CBAM attention (operates on original channels)
            self.cbam_layers.append(CBAM(in_c))
            # 1×1 channel alignment
            self.align_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, target_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_channels),
                    nn.ReLU(inplace=True),
                )
            )
            # 3×3 smooth conv after spatial adaptation
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(target_channels, target_channels,
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(target_channels),
                    nn.ReLU(inplace=True),
                )
            )

    # ── spatial adaptation ────────────────────────────────────────────────────
    @staticmethod
    def _adapt(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            _, _, h, w = x.shape
            if h == target_h and w == target_w:
                return x
            elif h > target_h:                             # downsampling
                return F.adaptive_avg_pool2d(x, (target_h, target_w))
            else:                                          # upsampling
                # Ép kiểu x sang float32 để nội suy, sau đó ép ngược lại kiểu gốc (bf16/fp16)
                return F.interpolate(x.to(torch.float32), size=(target_h, target_w),
                                     mode='bilinear', align_corners=False).to(x.dtype)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features : 4 NCHW tensors from the encoder skip list,
                   ordered shallow (large) → deep (small).
        """
        assert len(features) == 4
        # Infer target spatial size from the map at `target_index`
        target_h = features[self.target_index].shape[2]
        target_w = features[self.target_index].shape[3]

        fused = None
        for i, (feat, cbam, align, smooth) in enumerate(
                zip(features, self.cbam_layers,
                    self.align_convs, self.smooth_convs)):
            # 1. CBAM attention
            f = cbam(feat)
            # 2. 1×1 channel alignment
            f = align(f)
            # 3. Spatial adaptation (pool / identity / interpolate)
            f = self._adapt(f, target_h, target_w)
            # 4. Smooth 3×3 conv
            f = smooth(f)
            # 5. Hadamard product accumulation
            fused = f if fused is None else fused * f

        return fused   # (B, target_channels, target_h, target_w)


#  3.  VSSM V2 — Backbone with SDI skip connections + deep supervision
class VSSM(nn.Module):
    """
    VM-UNetV2 backbone.

    Changes vs V1
    ─────────────
    • skip connections replaced by SDI modules (one per non-bottleneck
      decoder stage, i.e. inx = 1, 2, 3 in layers_up)
    • deep supervision: decoder stages 1, 2, 3 each produce an auxiliary
      segmentation head; forward() returns a list of 3 masks
      (index 0 = coarsest / deepest, index 2 = finest / shallowest)
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768],
                 dims_decoder=[768, 384, 192, 96],
                 d_state=16, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers  = len(depths)

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]
        self.embed_dim    = dims[0]
        self.num_features = dims[-1]
        self.dims         = dims

        # ── Encoder ───────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in
                       torch.linspace(0, drop_path_rate,
                                      sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):
                               sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if i_layer < self.num_layers - 1
                           else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):
                                       sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if i_layer != 0 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # ── SDI modules (stages 1, 2, 3 of the decoder) ──────────────────────
        #
        #   Decoder stage 0 (inx=0): bottleneck, no skip → no SDI
        #   Decoder stage 1 (inx=1): target resolution = enc stage 2 → target_index=2
        #   Decoder stage 2 (inx=2): target resolution = enc stage 1 → target_index=1
        #   Decoder stage 3 (inx=3): target resolution = enc stage 0 → target_index=0
        #
        #   dims_decoder[1]=384, [2]=192, [3]=96  after the upsample inside
        #   each VSSLayer_up, the tensor channel count equals dims_decoder[inx].
        #   The SDI output must match that so it can be fed into the layer.
        #
        #   Encoder feature channels  = dims = [96, 192, 384, 768]

        sdi_target_indices = [2, 1, 0]   # for decoder stages 1, 2, 3
        sdi_target_channels = [dims_decoder[1],   # 384
                                dims_decoder[2],   # 192
                                dims_decoder[3]]   # 96

        self.sdi_modules = nn.ModuleList([
            SDI(encoder_channels=dims,           # [96,192,384,768]
                target_channels=tc,
                target_index=ti)
            for tc, ti in zip(sdi_target_channels, sdi_target_indices)
        ])

        # ── Final upsampler + segmentation head ───────────────────────────────
        self.final_up   = Final_PatchExpand2D(dim=dims_decoder[-1],
                                              dim_scale=4,
                                              norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4,
                                    num_classes, kernel_size=1)

        # ── Deep supervision heads (decoder stages 1, 2, 3) ──────────────────
        #
        #   At each supervised stage the feature map is NHWC with channel
        #   dims_decoder[inx].  We permute to NCHW, then apply a light
        #   head: 1×1 conv → bilinear upsample to patch-grid size of that
        #   stage so all aux outputs share a common spatial size before
        #   the caller rescales them to the full image.
        #
        #   Stage 1 out: dims_decoder[1]=384,  H/16 × W/16  (enc stage 2 size)
        #   Stage 2 out: dims_decoder[2]=192,  H/8  × W/8   (enc stage 1 size)
        #   Stage 3 out: dims_decoder[3]=96,   H/4  × W/4   (enc stage 0 size)
        #
        #   Each head is a single 1×1 conv followed by bilinear ×up to make
        #   the aux logits the same spatial size as the stage-3 output
        #   (H/4 × W/4), which matches the final_conv output resolution
        #   (both are upsampled to full resolution in VMUNet).

        self.aux_heads = nn.ModuleList([
            nn.Conv2d(dims_decoder[1], num_classes, kernel_size=1),  # stage 1
            nn.Conv2d(dims_decoder[2], num_classes, kernel_size=1),  # stage 2
            nn.Conv2d(dims_decoder[3], num_classes, kernel_size=1),  # stage 3
        ])

        self.apply(self._init_weights)

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # ── Encoder pass ─────────────────────────────────────────────────────────
    def forward_features(self, x):
        """
        Returns
        -------
        x         : bottleneck feature  (B, H', W', C_deep)   NHWC
        skip_list : [f0, f1, f2, f3]   NHWC, shallow→deep
        """
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    # ── Decoder pass with SDI skip connections ────────────────────────────────
    def forward_features_up(self, x, skip_list):
        """
        Parameters
        ----------
        x         : bottleneck  (B, H', W', C_deep)  NHWC
        skip_list : list of 4 NHWC tensors (shallow → deep)

        Returns
        -------
        x        : final decoder feature  NHWC
        aux_outs : list of 3 NCHW tensors (coarse → fine) before sigmoid

        NOTE về thứ tự upsample + SDI
        ──────────────────────────────
        VSSLayer_up.forward() chạy upsample TRƯỚC rồi mới chạy VSS blocks.
        Vì vậy ta không thể cộng SDI trước khi gọi layer_up — kích thước
        spatial của x chưa được expand.

        Giải pháp: tách upsample ra ngoài (gọi trực tiếp layer_up.upsample),
        cộng SDI sau upsample, rồi chạy VSS blocks (layer_up.blocks).
        """
        aux_outs = []

        # Pre-compute NCHW versions của tất cả encoder skips cho SDI
        # (SDI nhận NCHW; Mamba layers dùng NHWC)
        skips_nchw = [f.permute(0, 3, 1, 2).contiguous()
                      for f in skip_list]   # [f0,f1,f2,f3] NCHW

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # ── Bottleneck: không có skip, không có SDI ───────────────
                # layer_up ở đây không có upsample (upsample=None khi inx==0)
                x = layer_up(x)
            else:
                # ── Bước 1: Upsample x trước (tách ra khỏi layer_up) ─────
                if layer_up.upsample is not None:
                    x = layer_up.upsample(x)   # NHWC, spatial ×2

                # ── Bước 2: SDI tạo skip connection đã căn chỉnh ─────────
                sdi_idx      = inx - 1
                sdi_out_nchw = self.sdi_modules[sdi_idx](skips_nchw)
                # (B, target_channels, H_t, W_t) NCHW

                # Convert NCHW → NHWC để cộng với x
                sdi_out_nhwc = sdi_out_nchw.permute(0, 2, 3, 1).contiguous()

                # Cộng SDI vào x (cả spatial + channel đã khớp)
                x = x + sdi_out_nhwc

                # ── Bước 3: Chạy VSS blocks ───────────────────────────────
                for blk in layer_up.blocks:
                    x = (checkpoint.checkpoint(blk, x)
                         if layer_up.use_checkpoint else blk(x))

                # ── Bước 4: Deep supervision head ─────────────────────────
                aux = x.permute(0, 3, 1, 2).contiguous()      # NCHW
                aux = self.aux_heads[sdi_idx](aux)             # (B, num_classes, H, W)
                aux_outs.append(aux)

        return x, aux_outs

    # ── Final projection to full resolution ───────────────────────────────────
    def forward_final(self, x):
        x = self.final_up(x)                       # NHWC, ×4 spatial
        x = x.permute(0, 3, 1, 2)                  # NCHW
        x = self.final_conv(x)                     # (B, num_classes, H, W)
        return x

    # ── Full forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        """
        Returns
        -------
        List of 4 logit tensors (all NCHW, all raw — no sigmoid):
            [aux_stage1, aux_stage2, aux_stage3, final_logits]

        • aux_stage1 : coarsest (H/16 × W/16)
        • aux_stage2 : mid      (H/8  × W/8)
        • aux_stage3 : fine     (H/4  × W/4)
        • final_logits: full    (H    × W  after VMUNet rescaling)

        Callers are responsible for upsampling aux logits to full resolution
        during training (e.g. with F.interpolate) and applying sigmoid /
        softmax when needed.
        """
        x, skip_list          = self.forward_features(x)
        x, aux_outs           = self.forward_features_up(x, skip_list)
        final_out             = self.forward_final(x)
        # aux_outs = [coarse, mid, fine]  (decoder stages 1,2,3)
        return aux_outs + [final_out]   # 4 tensors total
