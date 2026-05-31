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

Chỉ thêm 3 class nhỏ + 1 VSSM subclass.

Cách dùng:
    from vmunet_v3 import VMUNetSC
    model = VMUNetSC(input_channels=3, num_classes=1)
    model.load_from()   # load pretrained giống V1
"""

from .vmamba import VSSM
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Spatial Attention Branch ──────────────────────────────────────────────
class SpatialAttBridge(nn.Module):
    """
    Dùng shared conv để tạo spatial mask cho từng skip.
    Shared weights → các scale học cùng một "pattern" spatial.

    Input : t1..t4  NCHW (đã permute từ NHWC)
    Output: att_1..att_4  NCHW mask [0,1]
    """
    def __init__(self):
        super().__init__()
        # kernel 7, dilation 3 → receptive field rộng hơn conv 7×7 thông thường
        self.shared_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=9, dilation=3, bias=False),
            nn.Sigmoid()
        )

    def _att_one(self, t: torch.Tensor) -> torch.Tensor:
        avg = t.mean(dim=1, keepdim=True)          # (B,1,H,W)
        mx, _ = t.max(dim=1, keepdim=True)         # (B,1,H,W)
        return self.shared_conv(torch.cat([avg, mx], dim=1))

    def forward(self, t1, t2, t3, t4):
        return (self._att_one(t1), self._att_one(t2),
                self._att_one(t3), self._att_one(t4))


# ── 2. Channel Attention Branch ───────────────────────────────────────────────
class ChannelAttBridge(nn.Module):
    """
    Học tương quan channel CROSS-SCALE:
      - Pool mỗi skip → vector channel
      - Concat tất cả → Conv1d học quan hệ giữa các scale
      - Project ngược về channel riêng của từng skip

    c_list : [96, 192, 384, 768]  (dims của 4 encoder stages)
    """
    def __init__(self, c_list):
        super().__init__()
        c_sum = sum(c_list)                         # 96+192+384+768 = 1440

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Conv1d học quan hệ giữa các channel cross-scale
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # Project từ c_sum về channel riêng của từng skip
        self.proj = nn.ModuleList([
            nn.Linear(c_sum, c, bias=False) for c in c_list
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4):
        ts = [t1, t2, t3, t4]

        # Pool mỗi skip → (B, C_i, 1, 1) → squeeze → (B, C_i)
        pooled = [self.avg_pool(t).squeeze(-1).squeeze(-1) for t in ts]

        # Concat → (B, C_sum)
        cat = torch.cat(pooled, dim=1)

        # Conv1d cross-scale: (B, C_sum) → (B,1,C_sum) → conv → (B,1,C_sum) → (B,C_sum)
        cat_1d = self.conv1d(cat.unsqueeze(1)).squeeze(1)

        # Project + sigmoid → att mask mỗi skip
        atts = []
        for proj in self.proj:
            att = self.sigmoid(proj(cat_1d))        # (B, C_i)
            atts.append(att)

        # Expand về (B, C_i, 1, 1) để nhân với feature map
        return tuple(a.unsqueeze(-1).unsqueeze(-1) for a in atts)


# ── 3. SC_Att_Bridge = Spatial + Channel, theo thứ tự S→C ────────────────────
class SC_Att_Bridge(nn.Module):
    """
    Pipeline:
        1. Spatial attention (shared conv) → scale từng skip theo vị trí
        2. Residual cộng lại
        3. Channel attention (cross-scale Conv1d) → scale theo channel
        4. Residual cộng lại

    Kết quả: 4 skip đã được tinh chỉnh, "biết về nhau" trước khi vào decoder.
    """
    def __init__(self, c_list):
        super().__init__()
        self.satt = SpatialAttBridge()
        self.catt = ChannelAttBridge(c_list)

    def forward(self, t1, t2, t3, t4):
        # Lưu residual gốc
        r1, r2, r3, r4 = t1, t2, t3, t4

        # ── Spatial attention ─────────────────────────────────────────────────
        sa1, sa2, sa3, sa4 = self.satt(t1, t2, t3, t4)
        t1, t2, t3, t4 = sa1 * t1, sa2 * t2, sa3 * t3, sa4 * t4

        # Lưu sau spatial (dùng cho residual channel)
        s1, s2, s3, s4 = t1, t2, t3, t4

        # Residual 1: cộng lại skip gốc
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4

        # ── Channel attention cross-scale ─────────────────────────────────────
        ca1, ca2, ca3, ca4 = self.catt(t1, t2, t3, t4)
        t1, t2, t3, t4 = ca1 * t1, ca2 * t2, ca3 * t3, ca4 * t4

        # Residual 2: cộng lại sau spatial
        return t1 + s1, t2 + s2, t3 + s3, t4 + s4


# ── 4. VSSM subclass: override forward_features_up dùng SC_Att_Bridge ────────
class VSSM_SC(VSSM):
    """
    Kế thừa VSSM gốc, override đúng 1 method: forward_features_up.

    Thay đổi duy nhất:
        V1: x = layer_up(x + skip_list[-inx])          ← cộng thẳng 1 skip
        SC: skip đã qua SC_Att_Bridge trước             ← 4 skip "biết nhau"
            x = layer_up(x + refined_skip[-inx])
    """
    def __init__(self, sc_bridge: SC_Att_Bridge, **kwargs):
        super().__init__(**kwargs)
        self.sc_bridge = sc_bridge

    def forward_features_up(self, x, skip_list):
        """
        skip_list: [s0, s1, s2, s3]  NHWC, shallow→deep
            s0: (B, H/4,  W/4,  96)
            s1: (B, H/8,  W/8,  192)
            s2: (B, H/16, W/16, 384)
            s3: (B, H/32, W/32, 768)

        SC_Att_Bridge nhận NCHW → permute trước, bridge, permute lại.
        """
        # Permute NHWC → NCHW cho bridge
        s0, s1, s2, s3 = [s.permute(0, 3, 1, 2).contiguous()
                          for s in skip_list]

        # Cross-scale attention: tất cả 4 skip cùng lúc
        s0, s1, s2, s3 = self.sc_bridge(s0, s1, s2, s3)

        # Permute NCHW → NHWC để dùng trong decoder (Mamba dùng NHWC)
        refined = [s.permute(0, 2, 3, 1).contiguous()
                   for s in (s0, s1, s2, s3)]

        # Decoder loop — giống V1 nhưng dùng refined skip
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + refined[-inx])   # refined thay vì skip_list gốc

        return x


# ── 5. VMUNetSC — wrapper chính ───────────────────────────────────────────────
class VMUNet(nn.Module):
    """
    VMUNet + SC_Att_Bridge skip connection.

    API giống hệt VMUNet V1:
        model = VMUNetSC(input_channels=3, num_classes=1)
        model.load_from()
        logits = model(x)
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

        # SC_Att_Bridge dùng dims của 4 encoder stages
        sc_bridge = SC_Att_Bridge(c_list=list(dims))

        # VSSM_SC: VSSM gốc + bridge injected
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
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return logits

    def load_from(self):
        """Load pretrained VMamba checkpoint — logic giống hệt VMUNet V1."""
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
