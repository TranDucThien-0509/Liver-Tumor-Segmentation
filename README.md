# VM-UNetV2 — Project Structure

```
VM-UNet-V2/
│
├── VMUNetV2_Train.ipynb          ← Notebook chính (chỉ điều phối, không có logic)
│
├── configs/
│   ├── __init__.py
│   └── config.py                 ← TẤT CẢ hyperparameter & đường dẫn ở đây
│
├── models/
│   ├── __init__.py
│   └── vmunet/
│       ├── __init__.py
|       ├── vmamba_v1.py
|       ├── vmunet_v1.py    
│       ├── vmamba_v2.py          ← VSSM + CBAM + SDI + Deep Supervision
│       └── vmunet_v2.py          ← VMUNet wrapper + DeepSupervisionLoss helper
│
├── data/
│   ├── __init__.py
│   └── dataset.py                ← DataBlock, patient-level splitter, label_func
│
├── dataset/
│   ├── test
│   │   ├── images
│   │   └── masks
│   └── train
│       ├── images
│       └── masks
│
├── pre_trained_weights/
│   └── vmamba_small_e238_ema.pth ← File pre_trained_weights của bài báo gốc
│
├── preparation/
│   ├── Convert_3D_to_Slices.ipynb
│   └── Convert_dicom_to_nifties.ipynb
│
├── utils/
│   ├── __init__.py
│   ├── losses.py                 ← MultiClassDiceLoss, CombinedLoss, DeepSupervisionLoss
│   ├── metrics.py                ← foreground_acc, DSAwareDiceMulti
│   ├── checkpoint.py             ← Load pretrained VMamba weights
│   └── visualize.py              ← show_sample, find_tumor_images, show_predictions
│
└── scripts/
    ├── __init__.py
    ├── build_model.py            ← Factory: tạo VMUNet V2 từ Config
    ├── evaluation.py             ← So sánh model v1 với v2, đánh giá trên nhiều metrics
    ├── train.py                  ← Pipeline huấn luyện đầy đủ
    └── predict.py                ← Inference + visualize

```

## Cách sử dụng

### Chạy training đầy đủ (script):
```bash
cd /workspace/VM-UNet-V2
python scripts/train.py
```

### Chạy trong notebook:
Mở `VMUNetV2_Train.ipynb` — chỉ cần sửa **Section 2 (Config)** là đủ.

### Override nhanh từ notebook:
```python
from configs import Config, PathConfig, DataConfig, TrainConfig
cfg = Config(...)          # tuỳ chỉnh
from scripts.train import train
learn = train(cfg)
```

## Thay đổi thường gặp

| Muốn thay đổi | Sửa ở |
|---|---|
| Đường dẫn dữ liệu / weights | `configs/config.py` → `PathConfig` |
| Batch size, image size | `configs/config.py` → `DataConfig` |
| Kiến trúc model (depths, dims) | `configs/config.py` → `ModelConfig` |
| Số epochs, LR, weight decay | `configs/config.py` → `TrainConfig` |
| Trọng số deep supervision | `configs/config.py` → `TrainConfig.ds_weights` |
| Augmentation | `configs/config.py` → `DataConfig` (do_flip, max_rotate...) |
| Thêm class mới | `DataConfig.num_classes` + `DataConfig.class_names` |

## Deep Supervision

VM-UNetV2 ở chế độ **training** trả về list 4 tensors:
```
[aux1 (H/16), aux2 (H/8), aux3 (H/4), final (H)]
```
`DeepSupervisionLoss` tự động xử lý cả 2 trường hợp:
- Training → nhận list, tính weighted sum loss
- Inference → nhận tensor, tính loss bình thường
