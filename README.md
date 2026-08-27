# WDR-MoE

## Frequency-Aware Mixture-of-Experts for Fracture Detection in AI-Assisted Medical Imaging

[繁體中文](README.zh-TW.md)

This repository is the official implementation of **WDR-MoE**, a Wavelet-based Dynamic Routing Mixture-of-Experts framework for fracture detection. WDR-MoE extends [YOLO-Master](https://github.com/Tencent/YOLO-Master) with a Wavelet Frequency Expert (WFE) and a Frequency-aware Dynamic Router (FDR) to preserve subtle fracture cues and route features to suitable frequency experts.

## Highlights

- **Wavelet Frequency Expert (WFE):** decomposes feature maps into LL, LH, HL, and HH Haar sub-bands and learns frequency-specific representations.
- **Frequency-aware Dynamic Router (FDR):** uses global context and sparse top-k routing to select relevant experts for each input.
- **Cross-modality evaluation:** evaluated on RibFrac CT slices and multi-site FracAtlas radiographs.
- **Reproducible release:** includes model configurations, training code, dataset templates, curated metrics, and publication figures without redistributing source patient datasets.

## Architecture

<p align="center">
  <img src="img/WDR-MoE.jpg" width="900" alt="Overview of the WDR-MoE architecture">
</p>

WDR-MoE augments the YOLO-Master backbone with frequency-aware expert processing. The FDR predicts sparse routing weights, while fixed-Haar WFE branches retain low- and high-frequency fracture evidence before weighted feature fusion.

## Methodology

### Wavelet Frequency Expert (WFE)

<p align="center">
  <img src="img/WFE.jpg" width="820" alt="Wavelet Frequency Expert">
</p>

WFE applies a non-trainable 2D Haar discrete wavelet transform and separates the input into LL, LH, HL, and HH sub-bands. Each expert specializes in one band using pointwise and depthwise convolution, followed by nearest-neighbor upsampling. This release intentionally contains only the fixed-wavelet design used in the published experiments.

### Frequency-aware Dynamic Router (FDR)

<p align="center">
  <img src="img/FDR.jpg" width="820" alt="Frequency-aware Dynamic Router">
</p>

FDR uses globally pooled context to produce sparse top-k routing probabilities. The selected WFE outputs are fused with their routing weights, allowing the model to adapt its frequency response to each image.

## Repository layout

```text
.
├── assets/results/              # Curated metrics and aggregate plots
├── configs/                     # Dataset YAML templates
├── environment.yml              # Reproducible Conda environment
├── img/                         # Architecture and qualitative figures
├── ultralytics/
│   ├── cfg/models/master/       # Baseline and published WDR-MoE definitions
│   └── nn/modules/moe/          # Routers, experts, losses, and analysis tools
├── tools/                        # Prediction visualization utility
├── train.py                     # Reproducible training entry point
├── val.py                       # Reproducible validation entry point
└── test_wavelet_moe.py          # Fixed-Haar WFE smoke test
```

## Installation with Conda

Clone the repository and create the provided Python 3.10 Conda environment:

```bash
git clone https://github.com/Ryan-camak/WDR-MoE.git
cd WDR-MoE
conda env create -f environment.yml
conda activate wdr-moe
```

The environment file installs this repository and its required dependencies in editable mode. Verify the installation before preparing a dataset:

```bash
python -c "import torch; from ultralytics import YOLO; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python test_wavelet_moe.py
```

The default dependency resolver installs a compatible PyTorch package. If a specific CUDA build is required, create and activate the environment first, install the matching PyTorch build using the command from the [official PyTorch selector](https://pytorch.org/get-started/locally/), and then run `python -m pip install -e .` again inside `wdr-moe`. Do not install a separate global Ultralytics package, because this repository contains the modified WDR-MoE implementation.

## Datasets

Datasets are not redistributed here. Download them from their official sources and comply with their respective licenses and access requirements.

- **RibFrac:** the paper follows the official split and converts the 3D CT scans into axial 2D detection images. After retaining fracture-positive slices, the processed data contain 6,761 training images and 846 validation images.
- **FracAtlas:** the paper uses a 6:2:2 train/validation/test split.

Copy the corresponding template and edit its `path`:

```bash
cp configs/ribfrac.example.yaml configs/ribfrac.yaml
# or
cp configs/fracatlas.example.yaml configs/fracatlas.yaml
```

Labels use YOLO detection format: `class x_center y_center width height`, normalized to `[0, 1]`.

## Training

The default command reproduces the WDR-MoE configuration and principal hyperparameters reported in the paper: 200 epochs, 640 × 640 input, batch size 32, SGD, and cosine learning-rate decay from 0.005 to 0.00005.

```bash
python train.py --data configs/ribfrac.yaml --device 0
```

Common overrides:

```bash
python train.py \
  --data configs/ribfrac.yaml \
  --model ultralytics/cfg/models/master/v0_1/det/yolo-master-n-fixwavelet.yaml \
  --epochs 200 --batch 32 --imgsz 640 --moe-loss 0.15
```

The released model follows the paper exactly: fixed 2D Haar DWT, one frequency band per expert, and nearest-neighbor upsampling after frequency-specific processing.

## Validation and inference

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
metrics = model.val(data="configs/ribfrac.yaml", imgsz=640)
predictions = model.predict(source="path/to/images", imgsz=640, save=True)
```

Equivalent validation command:

```bash
python val.py --model path/to/best.pt --data configs/ribfrac.yaml --device 0
```

Run the wavelet analysis/reconstruction smoke test with:

```bash
python test_wavelet_moe.py
```

## Results

### RibFrac

| Method | Precision | Recall | mAP50 | F1-score |
|---|---:|---:|---:|---:|
| YOLOv11 | 74.7% | 57.9% | 63.0% | 65.2% |
| YOLOv12 | 76.4% | 59.1% | 64.7% | 66.7% |
| YOLO-Master | 76.9% | 60.6% | 64.6% | 67.8% |
| YOLOv5-CSFF | 71.4% | 55.8% | 59.1% | 62.7% |
| WCAY | 78.2% | 55.2% | 61.5% | 64.7% |
| **WDR-MoE (ours)** | **80.5%** | **61.7%** | **67.7%** | **69.9%** |

### FracAtlas

| Method | Precision | Recall | mAP50 | F1-score |
|---|---:|---:|---:|---:|
| YOLOv11 | 61.4% | 32.2% | 36.8% | 42.2% |
| YOLOv12 | 45.3% | 30.7% | 30.2% | 36.6% |
| YOLO-Master | 43.6% | 35.0% | 36.7% | 38.8% |
| YOLOv5-CSFF | 35.5% | 32.8% | 29.8% | 34.1% |
| WCAY | 48.4% | **39.3%** | 39.5% | 43.4% |
| **WDR-MoE (ours)** | **62.6%** | 34.4% | **40.9%** | **44.4%** |

### Ablation study on FracAtlas

| FDR | WFE | Precision | Recall | mAP50 | F1-score |
|:---:|:---:|---:|---:|---:|---:|
|  |  | 45.3% | 30.7% | 30.2% | 36.6% |
| ✓ |  | 57.6% | 31.1% | 36.3% | 40.4% |
|  | ✓ | 44.8% | 31.1% | 31.0% | 36.7% |
| ✓ | ✓ | **62.6%** | **34.4%** | **40.9%** | **44.4%** |

### Qualitative comparison

<p align="center">
  <img src="img/qualitative-results.jpg" width="900" alt="Qualitative fracture detection comparison on RibFrac and FracAtlas">
</p>

The top row shows RibFrac CT examples and the bottom row shows FracAtlas radiographs. Red boxes indicate detected fracture regions across the compared detectors.

### Released training artifacts

<p align="center">
  <img src="assets/results/ribfrac/wdr_moe/results.png" width="760" alt="WDR-MoE RibFrac training curves">
</p>

See [`assets/results`](assets/results) for released CSV files and aggregate plots. Checkpoints are omitted from Git history; selected weights should be distributed through GitHub Releases or Zenodo.

## Citation

Please cite the published paper. The supplied manuscript establishes the title and author order below; replace the venue and DOI placeholders with the publisher's final metadata before release.

```bibtex
@inproceedings{jhong2026wdrmoe,
  title     = {{WDR-MoE}: Frequency-Aware Mixture-of-Experts for Fracture Detection in AI-Assisted Medical Imaging},
  author    = {Jhong, Sin-Ye and Lin, Bo-Xian and Li, Shang-Lin and Chen, Yung-Yao and Luo, Huai-An},
  booktitle = {TODO: Official publication venue},
  year      = {2026},
  doi       = {TODO: Publisher DOI}
}
```

Please also cite the upstream YOLO-Master and Ultralytics projects as required by their repositories.

## License and acknowledgements

This fork retains the [AGPL-3.0 license](LICENSE) of its upstream code. It is based on [Tencent YOLO-Master](https://github.com/Tencent/YOLO-Master) and [Ultralytics](https://github.com/ultralytics/ultralytics). WDR-MoE also draws on the Haar wavelet downsampling literature and the MoE routing design discussed in the paper.

This research software is not a certified medical device and must not be used for clinical diagnosis without independent validation and regulatory review.
