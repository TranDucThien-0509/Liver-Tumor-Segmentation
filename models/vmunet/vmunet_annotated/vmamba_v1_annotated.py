# =============================================================================
# vmamba_v1.py  —  VMamba backbone cho bài toán phân vùng ảnh (image segmentation)
#
# Kiến trúc tổng quan (U-Net style):
#
#   Ảnh đầu vào  (B, 3, H, W)
#       │
#   PatchEmbed2D          ─── chia ảnh thành patch, chiếu thành embedding
#       │
#   Encoder (layers)      ─── 4 VSSLayer, mỗi tầng: N × VSSBlock + PatchMerging2D
#       │  └─ lưu skip connections (skip_list)
#   Bottleneck
#       │
#   Decoder (layers_up)   ─── 4 VSSLayer_up, mỗi tầng: PatchExpand2D + N × VSSBlock
#       │  └─ cộng skip connections từ encoder
#   Final_PatchExpand2D   ─── upsample ×4 về resolution gốc
#       │
#   Conv2D 1×1            ─── ra segmentation map (B, num_classes, H, W)
#
# Nền tảng lý thuyết:
#   - VMamba / Mamba SSM: State Space Model thay thế Self-Attention
#   - SS2D: quét ảnh theo 4 hướng để nắm phụ thuộc không gian dài hạn
#   - Cấu trúc encoder-decoder giống Swin-UNet nhưng dùng SSM thay Transformer
# =============================================================================

import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Thử import selective_scan từ thư viện mamba_ssm (bản CUDA chính thức)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# Thử import bản thay thế (không cần causal_conv1d)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# Ghi đè __repr__ để in DropPath đẹp hơn khi debug
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =============================================================================
# Hàm đếm FLOPs của selective scan (dùng để phân tích độ phức tạp tính toán)
# =============================================================================
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    Ước tính số phép tính MAC (Multiply-Accumulate) của bước selective scan.
    Chỉ dùng để benchmark/so sánh, không chạy trong quá trình training.

    Ký hiệu tensor:
        u      : (B, D, L)   — input sequence
        delta  : (B, D, L)   — time step
        A      : (D, N)      — ma trận chuyển trạng thái (state transition)
        B      : (B, N, L)   — ma trận input projection
        C      : (B, N, L)   — ma trận output projection
        D      : (D,)        — skip connection parameter
        z      : (B, D, L)   — gating vector (optional)
    """
    import numpy as np
    
    # Hàm tính FLOPs cho một phép einsum bằng cách dùng numpy để lấy số liệu
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # Chia 2 vì ta đếm MAC (multiply+add = 1 flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0

    # ── Bước 1: Tính deltaA = exp(delta ⊗ A) ─────────────────────────────────
    # einsum "bdl,dn->bdln": delta (B,D,L) × A (D,N) → (B,D,L,N)
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")

    # ── Bước 2: Tính deltaB_u = delta ⊗ B ⊗ u ───────────────────────────────
    if with_Group:
        # Trường hợp B là biến (variable B, grouped): (B,D,L) × (B,N,L) × (B,D,L)
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        # Trường hợp B không grouped: (B,D,L) × (B,D,N,L) × (B,D,L)
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    # ── Bước 3: Vòng lặp recurrent (L bước) ─────────────────────────────────
    # Mỗi bước: x = deltaA * x + deltaB_u, y = C^T x
    in_for_flops = B * D * N   # phép nhân deltaA * x
    if with_Group:
        # y = einsum("bdn,bdn->bd"): output projection
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops   # nhân với số bước L

    # ── Bước 4 (tuỳ chọn): Skip connection D và gating Z ─────────────────────
    if with_D:
        flops += B * D * L   # out = y + u * D
    if with_Z:
        flops += B * D * L   # out = out * silu(z)
    
    return flops


# =============================================================================
# PatchEmbed2D — Chia ảnh thành các patch và chiếu thành embedding
# =============================================================================
class PatchEmbed2D(nn.Module):
    r"""
    Chuyển ảnh đầu vào thành chuỗi patch embedding.

    Cơ chế:
        - Dùng Conv2D với kernel_size = stride = patch_size
          → giống như chia ảnh thành lưới patch không chồng lấp
        - Ví dụ: ảnh 224×224, patch_size=4 → feature map 56×56×embed_dim
        - Output NHWC (channel ở cuối) để phù hợp với Mamba layers

    Args:
        patch_size (int): Kích thước mỗi patch. Default: 4.
        in_chans   (int): Số channel ảnh đầu vào. Default: 3.
        embed_dim  (int): Số chiều embedding. Default: 96.
        norm_layer      : Lớp chuẩn hóa. Default: None.
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # Conv2D stride=patch_size: không chồng lấp, giảm resolution ÷ patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x).permute(0, 2, 3, 1)   # (B, H/p, W/p, embed_dim) — NHWC
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# PatchMerging2D — Downsample trong Encoder (giảm resolution ÷2, tăng channel ×2)
# =============================================================================
class PatchMerging2D(nn.Module):
    r"""
    Gộp 4 pixel lân cận thành 1, tăng chiều channel.
    Đây là thao tác downsampling trong encoder, tương tự strided convolution.

    Cơ chế:
        1. Lấy 4 pixel theo dạng checkerboard 2×2: x0(even,even), x1(odd,even),
           x2(even,odd), x3(odd,odd)
        2. Concat theo channel: (B, H/2, W/2, 4C)
        3. LayerNorm + Linear: 4C → 2C

    Kết quả: spatial giảm ÷2, channel tăng ×2
    (Ví dụ: stage 0 → stage 1: 56×56×96 → 28×28×192)

    Args:
        dim        (int): Số channel đầu vào C.
        norm_layer      : Lớp chuẩn hóa. Default: nn.LayerNorm.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)   # chiếu 4C → 2C
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # Xử lý trường hợp H hoặc W lẻ (không chia hết cho 2)
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # Lấy 4 pixel theo pattern checkerboard
        x0 = x[:, 0::2, 0::2, :]  # pixel (even_row, even_col) → (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # pixel (odd_row,  even_col) → (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # pixel (even_row, odd_col)  → (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # pixel (odd_row,  odd_col)  → (B, H/2, W/2, C)

        # Cắt bỏ phần dư nếu kích thước lẻ
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)
        x = x.view(B, H//2, W//2, 4 * C)

        x = self.norm(x)        # LayerNorm trên 4C channels
        x = self.reduction(x)   # Linear 4C → 2C

        return x   # (B, H/2, W/2, 2C)
    

# =============================================================================
# PatchExpand2D — Upsample trong Decoder (tăng resolution ×2, giảm channel ÷2)
# =============================================================================
class PatchExpand2D(nn.Module):
    """
    Tăng resolution spatial lên ×2 bằng cách "tách" channel thành pixel.
    Ngược với PatchMerging2D.

    Cơ chế:
        1. Linear: C → dim_scale² × C  (phình rộng channel)
        2. rearrange: tách channel thành pixel theo không gian
           "b h w (p1 p2 c) -> b (h p1) (w p2) c"
           → spatial tăng ×dim_scale, channel giảm ÷dim_scale²

    Kết quả: spatial ×2, channel ÷2
    (Ví dụ: 14×14×384 → 28×28×192)

    Args:
        dim       (int): Số channel đầu vào (SAU khi đã ×2, xem self.dim=dim*2).
        dim_scale (int): Hệ số upsample. Default: 2.
        norm_layer     : Lớp chuẩn hóa. Default: nn.LayerNorm.
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2            # dim thực sự của input (vì decoder dim list bị dịch)
        self.dim_scale = dim_scale
        # Linear: dim*2 → dim_scale² × dim*2 = 4 × dim*2
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)   # (B, H, W, dim_scale² × C)

        # Tách channel thành pixel: phân phối channel ra theo không gian
        # p1×p2 pixel mới = 1 pixel cũ, mỗi pixel mới có C//dim_scale channels
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)

        return x   # (B, H*dim_scale, W*dim_scale, C//dim_scale)
    

# =============================================================================
# Final_PatchExpand2D — Upsample lần cuối ×4 về resolution gốc
# =============================================================================
class Final_PatchExpand2D(nn.Module):
    """
    Upsample ×4 để đưa feature map từ H/4 × W/4 về H × W (resolution gốc).
    Tương tự PatchExpand2D nhưng dim_scale=4.

    Cần thiết vì PatchEmbed2D đã thu nhỏ ảnh ×4 lúc đầu (patch_size=4),
    nên bước này bù lại kích thước để ra segmentation map đúng resolution.

    Args:
        dim       (int): Số channel đầu vào.
        dim_scale (int): Hệ số upsample. Default: 4.
        norm_layer     : Lớp chuẩn hóa.
    """
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # Linear: dim → dim_scale² × dim = 16 × dim
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)   # (B, H, W, 16C)

        # Tách channel thành pixel 4×4 (16 pixel/vị trí cũ)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)

        return x   # (B, H*4, W*4, C//4)


# =============================================================================
# SS2D — 2D Selective State Space (trái tim của VMamba)
# =============================================================================
class SS2D(nn.Module):
    """
    Mở rộng Mamba SSM từ 1D sequence lên 2D ảnh bằng cách quét theo 4 hướng.

    Ý tưởng cốt lõi:
        Mamba gốc xử lý sequence 1D. Để áp dụng cho ảnh 2D, SS2D trải phẳng
        ảnh thành sequence theo 4 hướng khác nhau, chạy SSM độc lập trên từng
        hướng, rồi cộng 4 kết quả lại. Nhờ đó model học được phụ thuộc
        không gian theo mọi hướng.

    4 hướng quét:
        k=0: trái→phải, trên→dưới  (raster scan bình thường)
        k=1: trên→dưới, trái→phải  (transposed scan)
        k=2: phải→trái, dưới→trên  (flip của k=0)
        k=3: dưới→trên, phải→trái  (flip của k=1)

    Pipeline forward:
        x (B,H,W,C)
         → in_proj:    Linear C → 2×d_inner, tách thành x và z (gating signal)
         → conv2d:     Depthwise Conv 3×3 (local feature extraction)
         → forward_core: 4 hướng selective scan song song
         → cộng 4 output: y = y1 + y2 + y3 + y4
         → out_norm + gating: y = LayerNorm(y) × SiLU(z)
         → out_proj:   Linear d_inner → d_model

    Tham số SSM:
        d_model  : chiều input/output
        d_state  : chiều không gian trạng thái N (default 16)
        d_conv   : kernel size depthwise conv (default 3)
        expand   : hệ số mở rộng → d_inner = expand × d_model (default 2)
        dt_rank  : rank của time step projection (default d_model/16)
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)   # chiều nội tại (= 2×d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # ── Projection đầu vào: chiếu thành x (feature) và z (gating) ────────
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # ── Depthwise Conv2D: trích xuất đặc trưng cục bộ ────────────────────
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,    # depthwise: mỗi channel riêng biệt
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,   # padding để giữ nguyên kích thước
            **factory_kwargs,
        )
        self.act = nn.SiLU()   # activation sau conv

        # ── 4 Linear x_proj: chiếu thành (dt, B_ssm, C_ssm) cho mỗi hướng ──
        # dt_rank + d_state*2 = rank của time step + chiều B và C của SSM
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        # Gộp 4 weight matrix thành 1 Parameter để tính toán hiệu quả hơn
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, dt_rank+2*d_state, d_inner)
        del self.x_proj  # xoá 4 Linear gốc, chỉ giữ weight gộp

        # ── 4 Linear dt_proj: chiếu dt từ dt_rank → d_inner ─────────────────
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, d_inner, dt_rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))     # (K=4, d_inner)
        del self.dt_projs

        # ── Ma trận A_log và D cho cả 4 hướng ────────────────────────────────
        # A_logs: log của ma trận chuyển trạng thái, shape (K*D, N) = (4*d_inner, d_state)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        # Ds: skip connection parameter (scalar per dimension), shape (K*D,) = (4*d_inner,)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # Chọn kernel selective scan (v0 = mamba_ssm CUDA, v1 = selective_scan thay thế)
        self.forward_core = self.forward_corev0

        # ── Output projection ─────────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """
        Khởi tạo Linear chiếu time step dt với bias đặc biệt.

        Mục tiêu: sau khi softplus, bias nằm trong [dt_min, dt_max].
        Cách thực hiện:
            1. Sample dt ~ Uniform[dt_min, dt_max] (log space)
            2. Tính inv_softplus(dt) để khi softplus(bias) = dt
        Bias được đánh dấu _no_reinit để tránh bị ghi đè sau khi khởi tạo.
        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Khởi tạo weight để giữ phương sai ổn định
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Khởi tạo bias sao cho softplus(bias) ∈ [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Tính inverse softplus: x = y + log(1 - exp(-y)) khi y > 0
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Đánh dấu để _init_weights của VSSM không reset bias này
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """
        Khởi tạo log của ma trận chuyển trạng thái A theo S4D real initialization.

        A là ma trận diagonal: A[i] = [1, 2, ..., d_state]
        Lưu log(A) để tính A = exp(A_log) → đảm bảo A > 0 mọi lúc.
        Thêm prefix 'r' copies cho K=4 hướng quét.
        """
        # S4D real: A[d, n] = n (n từ 1 đến d_state)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # giữ fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)   # (copies, d_inner, d_state)
            if merge:
                A_log = A_log.flatten(0, 1)   # (copies*d_inner, d_state)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # không decay A (nó là hyperparameter của SSM)
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """
        Khởi tạo tham số D (skip connection trong SSM: output += D * input).
        Khởi tạo bằng 1 (identity skip).
        """
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True  # không decay D
        return D

    def forward_corev0(self, x: torch.Tensor):
        """
        Lõi tính toán selective scan theo 4 hướng (dùng mamba_ssm CUDA kernel).

        Input: x (B, C, H, W) — feature map NCHW

        Các bước:
        1. Tạo 4 sequence bằng cách quét ảnh theo 4 hướng
        2. Chiếu mỗi sequence thành (dt, B_ssm, C_ssm) qua x_proj
        3. Chiếu dt từ dt_rank → d_inner qua dt_proj
        4. Chạy selective_scan_fn (CUDA) trên 4 hướng song song
        5. Đảo ngược và transpose các sequence flipped để về đúng vị trí không gian

        Output: 4 tensor (y1, y2, y3, y4) shape (B, d_inner, L)
            y1: hướng ngang bình thường (k=0)
            y2: flip hướng ngang đã đảo (k=2, đã flip ngược về)
            y3: hướng dọc đã transpose về ngang (k=1, đã transpose)
            y4: flip hướng dọc đã xử lý  (k=3, đã flip + transpose)
        """
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W   # số vị trí spatial = độ dài sequence
        K = 4       # số hướng quét

        # ── Bước 1: Tạo 4 sequence ────────────────────────────────────────────
        # Hướng ngang (k=0): trải phẳng ảnh theo hàng → (B, C, H*W)
        # Hướng dọc  (k=1): transpose H,W trước rồi trải phẳng → (B, C, W*H)
        x_hwwh = torch.stack([
            x.view(B, -1, L),                                              # (B, C, L) hướng ngang
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L) # (B, C, L) hướng dọc
        ], dim=1).view(B, 2, -1, L)   # (B, 2, C, L)

        # 2 hướng flip (đảo chiều sequence)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, C, L)

        # ── Bước 2: Chiếu thành tham số SSM (dt, B, C) ───────────────────────
        # x_proj_weight: (K=4, dt_rank+2*d_state, d_inner)
        # einsum "b k d l, k c d -> b k c l": chiếu d_inner → dt_rank+2*d_state cho mỗi hướng
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)

        # Tách thành 3 phần: time step rank, B_ssm, C_ssm
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # Chiếu dt: dt_rank → d_inner
        # einsum "b k r l, k d r -> b k d l": projection cho từng hướng k
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # ── Bước 3: Chuẩn bị input cho selective scan ────────────────────────
        xs = xs.float().view(B, -1, L)           # (B, K*d_inner, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, K*d_inner, L)
        Bs = Bs.float().view(B, K, -1, L)        # (B, K, d_state, L)
        Cs = Cs.float().view(B, K, -1, L)        # (B, K, d_state, L)
        Ds = self.Ds.float().view(-1)            # (K*d_inner,)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K*d_inner, d_state), luôn âm
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (K*d_inner,)

        # ── Bước 4: Chạy Selective Scan (CUDA kernel) ─────────────────────────
        # selective_scan_fn xử lý K*d_inner "channels" song song
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,       # áp dụng softplus cho delta (time step)
            return_last_state=False,
        ).view(B, K, -1, L)   # (B, 4, d_inner, L)
        assert out_y.dtype == torch.float

        # ── Bước 5: Đảo ngược các hướng flip và transpose ────────────────────
        # Hướng k=2, k=3 là flip của k=0, k=1 → cần flip ngược lại
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # (B, 2, d_inner, L): hướng k=2 và k=3 đã flip về đúng thứ tự gốc

        # Hướng k=1 là transposed scan → cần transpose lại (H,W ↔ W,H)
        wh_y = torch.transpose(
            out_y[:, 1].view(B, -1, W, H),   # reshape L → (W, H) vì scan theo chiều dọc
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)   # (B, d_inner, L) đúng thứ tự H×W

        # Hướng k=3 = flip của k=1 → đã flip + cần transpose
        invwh_y = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H),
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)

        # Trả về 4 hướng, tất cả đã về đúng thứ tự không gian (B, d_inner, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_corev1(self, x: torch.Tensor):
        """
        Phiên bản thay thế của forward_corev0, dùng selective_scan_fn_v1
        (không cần thư viện causal_conv1d).

        Logic hoàn toàn giống forward_corev0, chỉ khác ở:
        - selective_scan_fn_v1 thay cho selective_scan_fn
        - Không truyền z=None (API hơi khác)
        """
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward pass đầy đủ của SS2D.

        Pipeline:
            x (B,H,W,C)
            → in_proj: chiếu C → 2×d_inner, tách thành feature x và gate z
            → conv2d: depthwise 3×3 trên feature x (trích xuất local context)
            → forward_core: 4-direction selective scan
            → cộng 4 output: y = y1+y2+y3+y4
            → reshape về (B,H,W,d_inner)
            → out_norm: LayerNorm
            → gating: y = y × SiLU(z)   ← gating mechanism quan trọng
            → out_proj: chiếu d_inner → d_model
        """
        B, H, W, C = x.shape

        # Chiếu đầu vào thành x (feature) và z (gate)
        xz = self.in_proj(x)   # (B, H, W, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # x: (B,H,W,d_inner), z: (B,H,W,d_inner)

        # Depthwise conv: NHWC → NCHW → conv → vẫn NCHW
        x = x.permute(0, 3, 1, 2).contiguous()   # NHWC → NCHW
        x = self.act(self.conv2d(x))               # (B, d_inner, H, W)

        # 4-direction selective scan
        y1, y2, y3, y4 = self.forward_core(x)     # mỗi cái: (B, d_inner, L)
        assert y1.dtype == torch.float32

        # Cộng 4 hướng (tất cả đã về đúng thứ tự không gian)
        y = y1 + y2 + y3 + y4   # (B, d_inner, L)

        # Reshape từ sequence về feature map 2D
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B,H,W,d_inner)
        y = self.out_norm(y)          # LayerNorm

        # Gating: nhân với SiLU(z) — cơ chế điều chỉnh output theo gate signal
        y = y * F.silu(z)

        # Projection về chiều output
        out = self.out_proj(y)   # (B, H, W, d_model)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# =============================================================================
# VSSBlock — Block cơ bản của VMamba (SS2D + residual connection)
# =============================================================================
class VSSBlock(nn.Module):
    """
    Block đơn giản: Pre-Norm + SS2D + DropPath + Residual.

    Công thức:
        output = input + DropPath(SS2D(LayerNorm(input)))

    Tương tự Transformer Block nhưng thay Self-Attention bằng SS2D.
    Pre-Norm (LayerNorm trước SS2D) giúp training ổn định hơn Post-Norm.

    Args:
        hidden_dim    (int):   Chiều đặc trưng.
        drop_path     (float): Xác suất stochastic depth (tắt ngẫu nhiên trong training).
        norm_layer         :   Lớp chuẩn hóa (default LayerNorm eps=1e-6).
        attn_drop_rate (float): Dropout trong SS2D.
        d_state       (int):   Chiều trạng thái SSM.
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)          # Pre-Norm
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)         # Stochastic depth

    def forward(self, input: torch.Tensor):
        # Residual: output = input + DropPath(SS2D(LN(input)))
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# =============================================================================
# VSSLayer — Một stage của Encoder (stack VSSBlock + optional PatchMerging2D)
# =============================================================================
class VSSLayer(nn.Module):
    """
    Một stage của encoder: stack N VSSBlock rồi optional downsample.

    Thứ tự thực hiện:
        VSSBlock × depth → [PatchMerging2D]

    Sử dụng gradient checkpointing (use_checkpoint=True) để tiết kiệm VRAM
    bằng cách tính lại activation thay vì lưu cache trong backward pass.

    Args:
        dim          (int):  Số channel đặc trưng.
        depth        (int):  Số VSSBlock trong stage này.
        attn_drop    (float): Dropout rate cho SS2D.
        drop_path    (float | list): Stochastic depth rate (list cho mỗi block).
        norm_layer        :  Lớp chuẩn hóa.
        downsample        :  Class PatchMerging2D hoặc None.
        use_checkpoint(bool): Dùng gradient checkpointing.
        d_state      (int):  Chiều trạng thái SSM.
    """
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Stack depth VSSBlock với drop_path riêng cho mỗi block
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        # Khởi tạo out_proj.weight với Kaiming uniform
        # LƯU Ý: khởi tạo này sẽ bị ghi đè bởi VSSM._init_weights sau đó
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # tách gradient để không ảnh hưởng seed
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        # Downsample ở cuối stage (PatchMerging2D hoặc None)
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # Chạy tuần tự qua N VSSBlock
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)  # tiết kiệm VRAM
            else:
                x = blk(x)
        
        # Downsample nếu không phải stage cuối cùng của encoder
        if self.downsample is not None:
            x = self.downsample(x)

        return x


# =============================================================================
# VSSLayer_up — Một stage của Decoder (optional PatchExpand2D + stack VSSBlock)
# =============================================================================
class VSSLayer_up(nn.Module):
    """
    Một stage của decoder: optional upsample rồi stack N VSSBlock.

    Thứ tự thực hiện:
        [PatchExpand2D] → VSSBlock × depth

    Khác với VSSLayer: upsample ở ĐẦU (trước blocks), không phải ở cuối.
    Điều này giúp các VSS blocks xử lý ở resolution cao hơn.

    Args: (tương tự VSSLayer nhưng upsample thay downsample)
        upsample: Class PatchExpand2D hoặc None (stage 0 không upsample).
    """
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        # Upsample ở đầu stage (None cho bottleneck stage 0)
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        # Upsample trước (nếu có): tăng resolution spatial ×2
        if self.upsample is not None:
            x = self.upsample(x)
        # Chạy tuần tự qua N VSSBlock
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


# =============================================================================
# VSSM — Model chính: U-Net với VMamba backbone
# =============================================================================
class VSSM(nn.Module):
    """
    VMamba U-Net: kiến trúc encoder-decoder dùng Visual State Space Model.

    Tương tự Swin-UNet nhưng thay Self-Attention bằng SS2D (Selective Scan 2D).

    Encoder:
        PatchEmbed2D → VSSLayer × 4 (mỗi tầng có PatchMerging2D để giảm resolution)
        Lưu skip connections sau mỗi tầng encoder (trước khi downsample).

    Decoder:
        VSSLayer_up × 4 (mỗi tầng có PatchExpand2D để tăng resolution)
        Cộng skip connection tương ứng từ encoder (như U-Net).

    Final:
        Final_PatchExpand2D (upsample ×4) → Conv1×1 → segmentation map

    Args:
        patch_size     (int):        Kích thước patch. Default: 4.
        in_chans       (int):        Số channel đầu vào. Default: 3.
        num_classes    (int):        Số class segmentation. Default: 1000.
        depths         (list[int]):  Số VSSBlock mỗi encoder stage. Default: [2,2,9,2].
        depths_decoder (list[int]):  Số VSSBlock mỗi decoder stage. Default: [2,9,2,2].
        dims           (list[int]):  Channel mỗi encoder stage. Default: [96,192,384,768].
        dims_decoder   (list[int]):  Channel mỗi decoder stage. Default: [768,384,192,96].
        d_state        (int):        Chiều SSM state. Default: 16.
        drop_path_rate (float):      Stochastic depth rate tổng. Default: 0.1.
    """
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        # Nếu dims là int, tự động tạo list theo công thức dims × 2^i
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]        # 96 — chiều embedding đầu tiên
        self.num_features = dims[-1]    # 768 — chiều bottleneck
        self.dims = dims

        # ── Patch Embedding ───────────────────────────────────────────────────
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # ── Absolute Position Embedding (tắt, để lại cho khả năng mở rộng) ───
        self.ape = False  # tắt APE — VMamba không dùng position embedding tuyệt đối
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── Stochastic Depth schedule ─────────────────────────────────────────
        # drop_path tăng dần từ 0 → drop_path_rate qua encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # drop_path giảm dần từ drop_path_rate → 0 qua decoder (đảo ngược)
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # ── Encoder stages ────────────────────────────────────────────────────
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                # Cắt dpr list lấy đúng phần cho stage này
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # Downsample tất cả trừ stage cuối (bottleneck)
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # ── Decoder stages ────────────────────────────────────────────────────
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                # Upsample tất cả trừ stage đầu (bottleneck)
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # ── Final upsample + segmentation head ───────────────────────────────
        # Upsample ×4 từ H/4×W/4 → H×W
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        # Conv 1×1: chiếu về num_classes
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

        # ── Khởi tạo tham số ─────────────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Khởi tạo tham số model:
        - Linear.weight: trunc_normal (std=0.02)
        - Linear.bias: zeros
        - LayerNorm.bias: zeros, .weight: ones

        LƯU Ý: out_proj.weight từ VSSBlock (Kaiming uniform) sẽ bị ghi đè ở đây
        bởi trunc_normal. Conv2D cũng KHÔNG được khởi tạo ở đây (potential bug).
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Bỏ qua absolute_pos_embed khỏi weight decay (nếu có APE)."""
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Bỏ qua relative_position_bias_table khỏi weight decay."""
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        """
        Encoder pass: tạo feature map đa tầng và lưu skip connections.

        Thứ tự:
            patch_embed → pos_drop → 4 VSSLayer (lưu skip trước mỗi tầng)

        Returns:
            x        : bottleneck feature (B, H/32, W/32, 768)  NHWC
            skip_list: [stage0, stage1, stage2, stage3]  NHWC
                stage0: (B, H/4,  W/4,  96)   — shallow, detail cao
                stage1: (B, H/8,  W/8,  192)
                stage2: (B, H/16, W/16, 384)
                stage3: (B, H/32, W/32, 768)  — deep, semantic cao
        """
        skip_list = []
        x = self.patch_embed(x)   # (B, H/4, W/4, 96)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)   # lưu TRƯỚC khi downsample
            x = layer(x)          # VSSBlock × depth + PatchMerging2D
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        """
        Decoder pass: kết hợp skip connections và upsample dần.

        Kết nối skip theo kiểu U-Net:
            inx=0: bottleneck — không cộng skip
            inx=1: x + skip_list[-1] = skip của stage 3 (sâu nhất)
            inx=2: x + skip_list[-2] = skip của stage 2
            inx=3: x + skip_list[-3] = skip của stage 1

        LƯU Ý: thứ tự này đảm bảo decoder stage nhận skip có cùng resolution
        (vì VSSLayer_up upsample ở ĐẦU stage).

        Returns:
            x: feature map cuối decoder (B, H/4, W/4, 96)  NHWC
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)              # bottleneck: không cộng skip
            else:
                x = layer_up(x+skip_list[-inx])  # cộng skip rồi đưa vào layer_up

        return x

    def forward_final(self, x):
        """
        Bước cuối: upsample ×4 và chiếu về số class.

        x (B, H/4, W/4, 96) NHWC
        → Final_PatchExpand2D: (B, H, W, 24) NHWC
        → permute: (B, 24, H, W) NCHW
        → Conv1×1: (B, num_classes, H, W)
        """
        x = self.final_up(x)            # (B, H, W, 96//4=24) NHWC
        x = x.permute(0, 3, 1, 2)       # NHWC → NCHW
        x = self.final_conv(x)          # (B, num_classes, H, W)
        return x

    def forward_backbone(self, x):
        """
        Chỉ chạy encoder (dùng khi extract features, không cần decode).
        Trả về bottleneck feature map.
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        """
        Full forward pass cho segmentation.

        Input:  x (B, 3, H, W)
        Output: logits (B, num_classes, H, W)

        Pipeline:
            1. forward_features  → encoder + skip_list
            2. forward_features_up → decoder với skip connections
            3. forward_final     → upsample ×4 + conv → segmentation map
        """
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x
