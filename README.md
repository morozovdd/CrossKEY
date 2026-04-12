<p align="center">
  <h1 align="center"><ins>CrossKEY</ins><br>A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=XO_2QcAAAAAJ">Daniil Morozov</a><sup>1,2</sup>
    &middot;
    <a href="https://scholar.google.com/citations?user=xdECLMkAAAAJ&hl=fr">Reuben Dorent</a><sup>3,4</sup>
    &middot;
    <a href="https://scholar.google.com/citations?user=PjpzomsAAAAJ&hl=fr&oi=ao">Nazim Haouchine</a><sup>2</sup>
  </p>
  <p align="center">
    <sup>1</sup> Technical University of Munich, &nbsp;<sup>2</sup> Harvard Medical School, Brigham and Women's Hospital, &nbsp;
    <sup>3</sup> Inria, &nbsp;
    <sup>4</sup> Sorbonne Universit&eacute;, Paris Brain Institute
  </p>
</p>

<div align="center">

<a href="https://arxiv.org/abs/2507.18551"><img src="https://img.shields.io/badge/arXiv-2507.18551-b31b1b" alt='arxiv'></a>
<a href="https://huggingface.co/spaces/morozovdd/CrossKEY"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow" alt='HF Space'></a>
<a href="https://colab.research.google.com/github/morozovdd/CrossKEY/blob/main/notebooks/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt='Open in Colab'></a>
<a href="https://www.cancerimagingarchive.net/collection/remind/"><img src="https://img.shields.io/badge/Dataset-ReMIND-blue" alt='Dataset'></a>
<a href="#license"><img src="https://img.shields.io/badge/License-MIT-green" alt='License'></a>

</div>

<p align="center">
  <img src="assets/main.gif" alt="CrossKEY Demo" width="80%">
</p>

## Abstract

Intraoperative registration of real-time ultrasound (iUS) to preoperative Magnetic Resonance Imaging (MRI) remains an unsolved problem due to severe modality-specific differences in appearance, resolution, and field-of-view. To address this, we propose a novel 3D cross-modal keypoint descriptor for MRI-iUS matching and registration. Our approach employs a **patient-specific matching-by-synthesis approach**, generating synthetic iUS volumes from preoperative MRI. This enables supervised contrastive training to learn a shared descriptor space. A **probabilistic keypoint detection strategy** is then employed to identify anatomically salient and modality-consistent locations. During training, a curriculum-based triplet loss with dynamic hard negative mining is used to learn descriptors that are i) robust to iUS artifacts such as speckle noise and limited coverage, and ii) rotation-invariant. At inference, the method detects keypoints in MR and real iUS images and identifies sparse matches, which are then used to perform rigid registration. Our approach is evaluated using 3D MRI-iUS pairs from the ReMIND dataset. Experiments show that our approach outperforms state-of-the-art keypoint matching methods across 11 patients, with an average precision of **69.8%**. For image registration, our method achieves a competitive mean Target Registration Error of **2.39 mm** on the ReMIND2Reg benchmark.

<p align="center">
  <img src="assets/Overview.png" alt="Method Overview" width="100%">
</p>

## Getting Started

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Linux or macOS (for SIFT3D compilation; macOS requires [Homebrew](https://brew.sh))

### Installation

```bash
# Clone (fast, skips large data files):
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/morozovdd/CrossKEY.git
cd CrossKEY
git lfs pull

# Install dependencies and build SIFT3D:
./setup.sh
source .venv/bin/activate
```

### Training

```bash
python example_train.py [--config configs/train_config.yaml] [--data-dir data]
```

On first run, SIFT descriptors and keypoint heatmaps are generated automatically. Checkpoints are saved to `logs/`.

### Testing

```bash
python example_test.py --checkpoint path/to/checkpoint.ckpt [--data-dir data]
```

## Data

### Included Example

The repository includes one case (Case059) for testing:

```
data/img/
├── mr/              # T2-weighted brain MRI (.nii.gz)
├── us/              # Real intraoperative ultrasound (.nii.gz)
└── synthetic_us/    # Synthetic US generated from MR (.nii.gz)
```

### Using Your Own Data

Place your NIfTI files (`.nii.gz`) in the same directory structure above. Requirements:

- **MR**: 3D brain MRI (T1/T2 weighted)
- **Synthetic US**: Generated from MR using an ultrasound synthesis pipeline (required for training)
- **Real US**: 3D intraoperative ultrasound volume (required for testing only)

SIFT descriptors and heatmaps are generated automatically on first training run.

## Configuration

Training and evaluation are configured via YAML files in `configs/`:

- `train_config.yaml` -- model architecture, loss, optimizer, data augmentation, training schedule
- `test_config.yaml` -- checkpoint path, evaluation thresholds

All parameters can also be overridden via command-line arguments. Run `python example_train.py --help` for details.

## Project Structure

```
CrossKEY/
├── src/
│   ├── model/
│   │   ├── descriptor.py     # Lightning module for descriptor learning
│   │   ├── networks.py       # 3D ResNet encoder
│   │   ├── losses.py         # Triplet, InfoNCE, BCE losses
│   │   └── matcher.py        # KNN matching and evaluation
│   ├── data/
│   │   ├── datamodule.py     # Lightning DataModule
│   │   ├── dataset.py        # Training and inference datasets
│   │   └── transforms.py     # 3D rotation, crop, normalization
│   └── utils/
│       ├── sift.py           # SIFT3D wrapper
│       └── utils.py          # NIfTI I/O utilities
├── scripts/
│   ├── run_sift.py           # SIFT3D keypoint extraction
│   └── create_heatmaps.py    # Probabilistic keypoint heatmaps
├── configs/                   # YAML configuration files
├── data/img/                  # Example data (Case059)
├── example_train.py           # Training entry point
└── example_test.py            # Evaluation entry point
```

## Citation

```bibtex
@ARTICLE{11474556,
  author={Morozov, Daniil and Dorent, Reuben and Haouchine, Nazim},
  journal={IEEE Transactions on Medical Imaging},
  title={A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Feeds;Speckle;Filtering;Filters;Optical noise;Circuits and systems;Communication systems;Digital images;Protocols;Spatial diversity;Cross-modality;3D Keypoint Descriptor;MRI;Ultrasound;Matching and Registration},
  doi={10.1109/TMI.2026.3680352}}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
