"""
vmunet_v3.py  —  VMUNet + SC_Att_Bridge skip connection
=========================================================
Cải tiến so với V1:
    Thay skip đơn giản  `x + skip_list[-inx]`
    bằng SC_Att_Bridge  xử lý TẤT CẢ 4 skip cùng lúc trước decoder.

Tại sao SC_Att_Bridge tốt hơn?
    V1  : mỗi decoder stage chỉ nhận 1 skip tương ứng, các skip không
          biết gì về nhau.
    SC   : spatial attention + channel attention học CROSS-SCALE —
          skip nông (detail) và skip sâu (semantic) tương tác nhau
          trước khi vào decoder, giúp decoder có context phong phú hơn.

So sánh với V2 (SDI):
    SDI  : mỗi decoder stage có SDI riêng → fuse khác nhau cho từng stage
           (decoder sâu nhận fuse theo resolution sâu, nông nhận theo resolution nông)
    SC   : 1 bridge xử lý tất cả 4 skip một lần → 4 refined skip dùng chung cho
           toàn bộ decoder, đơn giản và nhẹ hơn SDI

Cấu trúc:
    SpatialAttBridge  — shared conv tạo spatial mask cho mỗi skip
    ChannelAttBridge  — cross-scale Conv1d học quan hệ channel giữa các skip
    SC_Att_Bridge     — kết hợp Spatial → Channel theo thứ tự S→C
    VSSM_SC           — VSSM kế thừa, override forward_features_up dùng SC_Att_Bridge
    VMUNet            — wrapper chính (API giống V1)
"""

from .vmamba import VSSM
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Module 1: SpatialAttBridge — Spatial Attention dùng shared conv
# =============================================================================
class SpatialAttBridge(nn.Module):
    """
    Tạo spatial attention mask cho từng skip connection.

    Thiết kế "shared conv" — tất cả 4 skip dùng chung 1 conv.
    Lý do: các skip ở các scale khác nhau nhưng đều cần học
    cùng một "pattern" không gian (e.g., cạnh biên, vùng tập trung).
    Shared weights đẩy mạnh tính nhất quán giữa các scale.

    Conv đặc biệt:
        kernel=7, dilation=3, padding=9
        → effective receptive field: 7 + (7-1)×(3-1) = 19 pixels
        → rộng hơn nhiều so với conv 7×7 thông thường (7 pixels)
        → học được context không gian ở phạm vi rộng hơn

    Pipeline cho mỗi skip:
        t (B,C,H,W)
        → mean theo channel: (B,1,H,W) — thông tin "trung bình" mỗi vị trí
        → max  theo channel: (B,1,H,W) — thông tin "đỉnh"  mỗi vị trí
        → cat:  (B,2,H,W)
        → shared_conv → sigmoid: (B,1,H,W) — spatial mask [0,1]
    """
    def __init__(self):
        super().__init__()
        # Shared conv: 2 channel → 1 channel spatial mask
        # dilation=3 mở rộng receptive field mà không tăng params
        self.shared_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=9, dilation=3, bias=False),
            nn.Sigmoid()
        )

    def _att_one(self, t: torch.Tensor) -> torch.Tensor:
        """Tạo spatial mask cho 1 skip tensor."""
        avg = t.mean(dim=1, keepdim=True)    # (B,1,H,W) — avg theo channel
        mx, _ = t.max(dim=1, keepdim=True)   # (B,1,H,W) — max theo channel
        return self.shared_conv(torch.cat([avg, mx], dim=1))   # (B,1,H,W)

    def forward(self, t1, t2, t3, t4):
        """
        Tạo spatial mask cho tất cả 4 skip.

        Args:
            t1..t4: NCHW tensors (đã permute từ NHWC)
        Returns:
            tuple 4 spatial masks (B,1,H_i,W_i) — mỗi skip có kích thước riêng
        """
        return (self._att_one(t1), self._att_one(t2),
                self._att_one(t3), self._att_one(t4))


# =============================================================================
# Module 2: ChannelAttBridge — Channel Attention CROSS-SCALE
# =============================================================================
class ChannelAttBridge(nn.Module):
    """
    Học quan hệ channel GIỮA CÁC SCALE (cross-scale channel attention).

    Điểm khác biệt so với Channel Attention thông thường:
        - Thông thường: mỗi feature map học attention trên channel của chính nó
        - ChannelAttBridge: pool TẤT CẢ 4 skip → concat → học quan hệ chéo scale

    Nhờ đó channel của skip nông (detail) có thể "biết" về channel của skip sâu
    (semantic) và điều chỉnh theo → alignment tốt hơn khi cộng vào decoder.

    Pipeline:
        t1..t4 (NCHW)
        → AdaptiveAvgPool2d(1): (B, C_i, 1, 1) → squeeze → (B, C_i)
        → cat: (B, C_sum)  trong đó C_sum = 96+192+384+768 = 1440
        → Conv1d cross-scale: (B,1,C_sum) → conv → (B,1,C_sum) → (B, C_sum)
          (Conv1d trên dim channel, học quan hệ giữa các channel kề nhau)
        → 4 Linear: C_sum → C_i cho mỗi skip (project về channel riêng)
        → sigmoid: (B, C_i) attention mask
        → expand về (B, C_i, 1, 1) để broadcast nhân với feature map

    Args:
        c_list (list[int]): Channel của 4 encoder outputs, e.g. [96, 192, 384, 768].
    """
    def __init__(self, c_list):
        super().__init__()
        c_sum = sum(c_list)    # 96+192+384+768 = 1440

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Conv1d: học quan hệ giữa các channel cross-scale
        # Xem channel vector (B, C_sum) như một sequence 1D dài C_sum
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # Project từ C_sum về channel riêng của từng skip
        self.proj = nn.ModuleList([
            nn.Linear(c_sum, c, bias=False) for c in c_list
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4):
        """
        Tính cross-scale channel attention mask.

        Returns:
            tuple 4 masks (B, C_i, 1, 1) — broadcast-ready
        """
        ts = [t1, t2, t3, t4]

        # Global avg pool mỗi skip: (B, C_i, H_i, W_i) → (B, C_i)
        pooled = [self.avg_pool(t).squeeze(-1).squeeze(-1) for t in ts]

        # Concat tất cả: (B, C_sum)
        cat = torch.cat(pooled, dim=1)

        # Conv1d cross-scale: cần shape (B, 1, C_sum) cho Conv1d
        # → conv → (B, 1, C_sum) → squeeze → (B, C_sum)
        cat_1d = self.conv1d(cat.unsqueeze(1)).squeeze(1)

        # Project về channel riêng + sigmoid → attention mask
        atts = []
        for proj in self.proj:
            att = self.sigmoid(proj(cat_1d))   # (B, C_i)
            atts.append(att)

        # Expand để broadcast nhân với feature map (B, C_i, H_i, W_i)
        return tuple(a.unsqueeze(-1).unsqueeze(-1) for a in atts)


# =============================================================================
# Module 3: SC_Att_Bridge — Spatial + Channel theo thứ tự S→C
# =============================================================================
class SC_Att_Bridge(nn.Module):
    """
    SC_Att_Bridge = Spatial Attention → Channel Attention (nối tiếp).

    Pipeline đầy đủ:
        ┌── Input: t1, t2, t3, t4 (NCHW) ───────────────────────────────────┐
        │                                                                     │
        │  [Spatial Attention]                                                │
        │  sa_i = SpatialAttBridge(t_i)  → spatial mask (B,1,H,W)           │
        │  t_i  = sa_i × t_i             → scale từng vị trí spatial        │
        │                                                                     │
        │  [Residual 1]                                                       │
        │  t_i  = t_i + r_i              → r_i là input gốc                 │
        │                                                                     │
        │  [Channel Attention cross-scale]                                    │
        │  ca_i = ChannelAttBridge(t_i)  → channel mask (B,C_i,1,1)         │
        │  t_i  = ca_i × t_i             → scale từng channel               │
        │                                                                     │
        │  [Residual 2]                                                       │
        │  t_i  = t_i + s_i              → s_i là output sau spatial        │
        └── Output: t1, t2, t3, t4 (NCHW, đã được tinh chỉnh) ─────────────┘

    Lý do dùng 2 residual connection:
        - Residual 1: bảo toàn thông tin gốc sau spatial attention
        - Residual 2: bảo toàn thông tin spatial sau channel attention
        → tránh mất thông tin, học incremental refinement

    Args:
        c_list (list[int]): Channel của 4 encoder outputs, e.g. [96,192,384,768].
    """
    def __init__(self, c_list):
        super().__init__()
        self.satt = SpatialAttBridge()
        self.catt = ChannelAttBridge(c_list)

    def forward(self, t1, t2, t3, t4):
        """
        Tinh chỉnh 4 skip connections qua spatial rồi channel attention.

        Args:
            t1..t4: NCHW tensors (encoder skips đã permute)
        Returns:
            tuple 4 NCHW tensors đã refined
        """
        # Lưu input gốc cho residual
        r1, r2, r3, r4 = t1, t2, t3, t4

        # ── Bước 1: Spatial Attention ─────────────────────────────────────────
        sa1, sa2, sa3, sa4 = self.satt(t1, t2, t3, t4)
        # Scale từng vị trí spatial: vị trí quan trọng → giữ, không quan trọng → giảm
        t1, t2, t3, t4 = sa1 * t1, sa2 * t2, sa3 * t3, sa4 * t4

        # Lưu sau spatial (dùng cho residual 2)
        s1, s2, s3, s4 = t1, t2, t3, t4

        # ── Bước 2: Residual 1 — cộng lại input gốc ──────────────────────────
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4

        # ── Bước 3: Channel Attention cross-scale ────────────────────────────
        ca1, ca2, ca3, ca4 = self.catt(t1, t2, t3, t4)
        # Scale từng channel: channel quan trọng → giữ, không quan trọng → giảm
        t1, t2, t3, t4 = ca1 * t1, ca2 * t2, ca3 * t3, ca4 * t4

        # ── Bước 4: Residual 2 — cộng lại sau spatial ────────────────────────
        return t1 + s1, t2 + s2, t3 + s3, t4 + s4


# =============================================================================
# VSSM_SC — Kế thừa VSSM gốc, override duy nhất forward_features_up
# =============================================================================
class VSSM_SC(VSSM):
    """
    VSSM với SC_Att_Bridge thay thế skip connection đơn giản.

    Chỉ override 1 method: forward_features_up.
    Tất cả encoder, decoder layers, khởi tạo, v.v. kế thừa nguyên vẹn từ VSSM.

    So sánh forward_features_up:
        VSSM gốc (V1):
            x = layer_up(x + skip_list[-inx])
            → cộng thẳng 1 skip tương ứng

        VSSM_SC:
            skip_list → SC_Att_Bridge → 4 refined skips
            x = layer_up.upsample(x) + refined_skip[-inx]
            → dùng skip đã được tinh chỉnh cross-scale

    Args:
        sc_bridge (SC_Att_Bridge): Bridge module đã khởi tạo từ bên ngoài.
        **kwargs: Các tham số khác truyền vào VSSM gốc (in_chans, depths, v.v.).
    """
    def __init__(self, sc_bridge: SC_Att_Bridge, **kwargs):
        super().__init__(**kwargs)
        self.sc_bridge = sc_bridge

    def forward_features_up(self, x, skip_list):
        """
        Decoder pass với SC_Att_Bridge thay thế skip connections.

        skip_list: [s0, s1, s2, s3] NHWC, shallow→deep
            s0: (B, H/4,  W/4,  96)
            s1: (B, H/8,  W/8,  192)
            s2: (B, H/16, W/16, 384)
            s3: (B, H/32, W/32, 768)

        LƯU Ý về format:
            - SC_Att_Bridge nhận và trả về NCHW
            - Mamba decoder layers dùng NHWC
            → cần permute trước bridge, permute lại sau bridge

        Pipeline:
            1. Permute tất cả skip: NHWC → NCHW
            2. SC_Att_Bridge: tinh chỉnh cross-scale (4 skip → 4 refined)
            3. Permute refined skips: NCHW → NHWC
            4. Decoder loop (giống V1 nhưng dùng refined thay skip gốc)
        """
        # ── Bước 1: Permute NHWC → NCHW cho SC_Att_Bridge ────────────────────
        s0, s1, s2, s3 = [s.permute(0, 3, 1, 2).contiguous()
                          for s in skip_list]

        # ── Bước 2: SC_Att_Bridge — cross-scale spatial + channel attention ──
        # Tất cả 4 skip tương tác nhau cùng lúc
        s0, s1, s2, s3 = self.sc_bridge(s0, s1, s2, s3)

        # ── Bước 3: Permute NCHW → NHWC ──────────────────────────────────────
        refined = [s.permute(0, 2, 3, 1).contiguous()
                   for s in (s0, s1, s2, s3)]
        # refined[0]: s0 (H/4), refined[1]: s1 (H/8)
        # refined[2]: s2 (H/16), refined[3]: s3 (H/32)

        # ── Bước 4: Decoder loop ──────────────────────────────────────────────
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # Bottleneck: không cộng skip (giống V1)
                x = layer_up(x)
            else:
                # Cộng refined skip thay vì skip gốc
                # refined[-inx]: inx=1 → refined[-1]=s3(sâu), inx=2 → s2, inx=3 → s1
                x = layer_up(x + refined[-inx])

        return x


# =============================================================================
# VMUNet — Wrapper chính (API giống V1)
# =============================================================================
class VMUNet(nn.Module):
    """
    VMUNet V3: VMUNet + SC_Att_Bridge skip connection.

    API giống hệt VMUNet V1 — drop-in replacement:
        model = VMUNet(input_channels=3, num_classes=1)
        model.load_from()
        logits = model(x)   # training và inference đều ra 1 tensor duy nhất

    Khác V1 ở chỗ skip connections đi qua SC_Att_Bridge trước khi vào decoder.
    Khác V2 ở chỗ không có deep supervision và không có SDI per-stage.

    Args:
        input_channels (int):   Số channel đầu vào. Default: 3.
        num_classes    (int):   Số class segmentation. Default: 1.
        depths         (tuple): Số block mỗi encoder stage. Default: (2,2,9,2).
        depths_decoder (tuple): Số block mỗi decoder stage. Default: (2,9,2,2).
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        load_ckpt_path (str):   Đường dẫn pretrained checkpoint. Default: None.
        dims           (tuple): Channel của 4 encoder stages. Default: (96,192,384,768).
    """
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 1,
                 depths=(2, 2, 9, 2),
                 depths_decoder=(2, 9, 2, 2),
                 drop_path_rate: float = 0.2,
                 load_ckpt_path: str = None,
                 dims=(96, 192, 384, 768)):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes    = num_classes

        # Khởi tạo SC_Att_Bridge với dims của 4 encoder stages
        sc_bridge = SC_Att_Bridge(c_list=list(dims))

        # VSSM_SC: VSSM gốc + SC_Att_Bridge được inject vào
        # Dùng pattern Dependency Injection thay vì hardcode trong VSSM
        self.vmunet = VSSM_SC(
            sc_bridge      = sc_bridge,
            in_chans       = input_channels,
            num_classes    = num_classes,
            depths         = list(depths),
            depths_decoder = list(depths_decoder),
            drop_path_rate = drop_path_rate,
            dims           = list(dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (đơn giản như V1, không có deep supervision).

        Args:
            x: (B, C, H, W)
        Returns:
            sigmoid probabilities hoặc logits: (B, num_classes, H, W)
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)   # grayscale → RGB giả
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)  # binary
        return logits                     # multi-class

    def load_from(self):
        """
        Load pretrained VMamba checkpoint — logic giống hệt VMUNet V1.

        Bước 1 — Encoder: load trực tiếp các key khớp.
        Bước 2 — Decoder: remap layers.i → layers_up.(3-i).
        """
        if self.load_ckpt_path is None:
            return

        model_dict      = self.vmunet.state_dict()
        checkpoint      = torch.load(self.load_ckpt_path)
        pretrained_dict = checkpoint['model']

        # ── Encoder ───────────────────────────────────────────────────────────
        new_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict}
        model_dict.update(new_dict)
        print(f'Encoder — model: {len(model_dict)}, '
              f'pretrained: {len(pretrained_dict)}, matched: {len(new_dict)}')
        self.vmunet.load_state_dict(model_dict)
        not_loaded = [k for k in pretrained_dict if k not in new_dict]
        print('Not loaded (encoder):', not_loaded)
        print('Encoder loaded!')

        # ── Decoder (remap layers.i → layers_up.3-i) ─────────────────────────
        remap = {'layers.0': 'layers_up.3', 'layers.1': 'layers_up.2',
                 'layers.2': 'layers_up.1', 'layers.3': 'layers_up.0'}

        model_dict      = self.vmunet.state_dict()
        pretrained_dict = checkpoint['model']
        remapped = {}
        for k, v in pretrained_dict.items():
            for src, dst in remap.items():
                if src in k:
                    remapped[k.replace(src, dst)] = v
                    break

        new_dict = {k: v for k, v in remapped.items() if k in model_dict}
        model_dict.update(new_dict)
        print(f'Decoder — remapped: {len(remapped)}, matched: {len(new_dict)}')
        self.vmunet.load_state_dict(model_dict)
        not_loaded = [k for k in remapped if k not in new_dict]
        print('Not loaded (decoder):', not_loaded)
        print('Decoder loaded!')
