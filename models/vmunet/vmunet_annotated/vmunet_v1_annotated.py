# =============================================================================
# vmunet_v1.py  —  VMUNet V1: wrapper đơn giản nhất cho VSSM
#
# Vai trò:
#   - Bọc VSSM thành model segmentation hoàn chỉnh
#   - Xử lý ảnh grayscale (1 channel → lặp 3 lần để thành RGB)
#   - Áp sigmoid cho binary segmentation (num_classes=1)
#   - Load pretrained checkpoint, remap encoder weights sang decoder
# =============================================================================

from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    """
    VMUNet V1: model phân vùng ảnh dựa trên VSSM backbone.

    Đây là wrapper đơn giản nhất — không có deep supervision, không có
    attention module bổ sung. Phù hợp làm baseline.

    Args:
        input_channels (int):   Số channel ảnh đầu vào (1 hoặc 3). Default: 3.
        num_classes    (int):   Số class segmentation. Default: 1.
        depths         (list):  Số block mỗi encoder stage. Default: [2,2,9,2].
        depths_decoder (list):  Số block mỗi decoder stage. Default: [2,9,2,2].
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        load_ckpt_path (str):   Đường dẫn checkpoint pretrained VMamba. Default: None.
    """
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        # Khởi tạo VSSM backbone với các tham số mặc định
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):
        """
        Forward pass.

        Xử lý đặc biệt:
        - Ảnh grayscale (1 channel): lặp 3 lần theo channel dim
          để tương thích với backbone pretrained trên ảnh RGB
        - Binary segmentation (num_classes=1): áp sigmoid → xác suất [0,1]
        - Multi-class: trả về logit thô → áp softmax ở loss function

        Args:
            x: (B, C, H, W) với C=1 hoặc C=3
        Returns:
            logits hoặc sigmoid probabilities: (B, num_classes, H, W)
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)   # grayscale → RGB giả: (B,1,H,W) → (B,3,H,W)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)  # binary: ra xác suất [0,1]
        else:
            return logits                 # multi-class: raw logits
    
    def load_from(self):
        """
        Load pretrained VMamba checkpoint và khởi tạo cả encoder lẫn decoder.

        Chiến lược 2 bước:
        ─────────────────
        Bước 1 — Load encoder:
            Lọc checkpoint lấy các key khớp với model hiện tại.
            Cập nhật encoder weights trực tiếp.

        Bước 2 — Remap encoder → decoder:
            VMamba pretrained chỉ có encoder (layers.0..3), không có decoder.
            Ta khởi tạo decoder bằng cách "lật" encoder:
                layers.0  →  layers_up.3  (stage nông nhất → stage decode sâu nhất)
                layers.1  →  layers_up.2
                layers.2  →  layers_up.1
                layers.3  →  layers_up.0  (stage sâu nhất → bottleneck decode)

            Lý do: encoder và decoder có cùng cấu trúc VSS blocks, nên weight
            encoder có thể là điểm khởi đầu tốt cho decoder tương ứng.

        In ra số lượng key đã load để debug.
        """
        if self.load_ckpt_path is not None:
            # ── Bước 1: Load encoder weights ──────────────────────────────────
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']

            # Lọc: chỉ giữ key có trong cả pretrained và model hiện tại
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            # In các key không được load (để kiểm tra xem thiếu gì)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            # ── Bước 2: Remap encoder weights → decoder ───────────────────────
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}

            # Đổi tên key: layers.i → layers_up.(3-i) (lật thứ tự)
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

            # Lọc và cập nhật decoder
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
