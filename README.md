Medical Image Segmentation: Advanced Structural Enhancements in Vision Mamba UNet

This repository contains the official implementation of VM-UNet V2, introducing advanced structural enhancements to the pure State Space Model (SSM) architecture for medical image segmentation. It includes Method 1 (SDI + CBAM + Deep Supervision) and Method 2 (SC_Att_Bridge), engineered to bridge the semantic gap and enhance multi-scale feature interactions

```
VM-UNet/
│
├── VMUNetV2_Train.ipynb         ← Main orchestration notebook (execution framework only)
│
├── configs/
│   ├── __init__.py
│   └── config.py                ← Centralized configuration: ALL hyperparameters & paths
│
├── models/
│   ├── __init__.py
│   └── vmunet/
│       ├── __init__.py
│       ├── vmamba_v1.py
│       ├── vmunet_v1.py         ← Original VM-UNet baseline architecture
│       ├── vmamba_v2.py         ← Method 1 primitives: VSSM + CBAM + SDI modules
│       ├── vmunet_v2.py         ← Method 1 wrapper + Deep Supervision integration
│       └── vmunet_v3.py         ← Method 2 wrapper: VM-UNet with Spatial-Channel Attention Bridge
│
├── data/
│   ├── __init__.py
│   └── dataset.py               ← DataBlock pipeline, patient-level splitters, and label functions
│
├── dataset/                     ← Local dataset directory
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
├── pre_trained_weights/
│   └── vmamba_small_e238_ema.pth ← Pretrained backbone weights from the baseline model
│
├── preparation/
│   ├── Convert_3D_to_Slices.ipynb
│   └── Convert_dicom_to_nifties.ipynb
│
├── utils/
│   ├── __init__.py
│   ├── losses.py                ← MultiClassDiceLoss, CombinedLoss, DeepSupervisionLoss
│   ├── metrics.py               ← Evaluation metrics: foreground_acc, DSAwareDiceMulti
│   ├── checkpoint.py            ← Weight loaders for pre-trained VMamba states
│   └── visualize.py             ← Verification helpers: show_sample, find_tumor_images, show_predictions
│
└── scripts/
    ├── __init__.py
    ├── build_model.py           ← Model Factory: Instantiates requested VM-UNet variant from Config
    ├── evaluation.py            ← Evaluation suite: Comprehensive multi-metric test validation
    ├── train.py                 ← End-to-end training execution pipeline
    └── predict.py               ← Production inference and prediction visualization

```

Usage Guide
1. Environment Installation
Create the dedicated virtual environment and install the required foundational dependencies:
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install mamba-ssm[causal-conv1d] --no-build-isolation
```

2. Executing via Command Line Script
To launch the end-to-end training pipeline directly using the script subsystem, run:
```bash
cd /workspace/VM-UNet-V2
python scripts/train.py
```

3. Executing via Jupyter Notebook
Open and run VMUNet_Train.ipynb. The notebook acts as an execution framework; you can completely control the pipeline workloads solely by adjusting settings inside Section 2 (Config).

4. Dynamic Code Configuration Overrides
You can instantiate a custom configuration object and inject it directly into the execution script programmatically:
```python
from configs import Config, PathConfig, DataConfig, TrainConfig
from scripts.train import train

# Customize configuration parameters programmatically
custom_cfg = Config(...)

# Launch the training routine
learn = train(custom_cfg)
```

Common Configuration Overrides
All operational parameters should be managed within configs/config.py by targeting the corresponding configuration class scope:

| Target Modification | Configuration Scope |
|---|---|
| Data Paths / Pretrained Weights | `configs/config.py` → `PathConfig` |
| Batch size, image size | `configs/config.py` → `DataConfig` |
| Architecture Scale (Depths / Dims) | `configs/config.py` → `ModelConfig` |
| Total Epochs, LR, Weight Decay | `configs/config.py` → `TrainConfig` |
| Deep Supervision Loss Weights | `configs/config.py` → `TrainConfig.ds_weights` |
| Data Augmentation Parameters | `configs/config.py` → `DataConfig` (do_flip, max_rotate...) |
| Class Topology / Target Cardinality | `DataConfig.num_classes` + `DataConfig.class_names` |


Deep Supervision Mechanism
When configured in Training Mode, Method 1 of VM-UNet V2 utilizes auxiliary segmentation heads to generate an output array consisting of 4 distinct tensors scaled to various spatial resolutions:
```
[aux1 (H/16), aux2 (H/8), aux3 (H/4), final (H)]
```
`DeepSupervisionLoss`

The custom DeepSupervisionLoss module manages both execution paths automatically:  
Training Mode: Processes the array of predicted outputs, interpolates intermediate feature scales, and executes a weighted summary loss computation using your defined configuration parameters.  
Inference Mode: Detects standard singleton tensors automatically, isolates the fine-resolution final_prediction tensor, and measures standard objective criteria while bypassing raw auxiliary branches cleanly.  
