# CrossKEY Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all code bugs, switch to uv, update README with IEEE TMI publication, fix clone time via Git LFS + history rewrite, and respond to the HuggingFace issue.

**Architecture:** The repo is a PyTorch Lightning research framework for 3D cross-modal keypoint descriptors (MR-US matching). Changes are isolated to: dependency management (uv migration), bug fixes across src/, script fixes, README updates, and git history cleanup. No architectural changes needed.

**Tech Stack:** Python 3.12+, PyTorch, Lightning 2.x, uv, Git LFS

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Switch to uv, add missing deps (tqdm, scipy), fix PEP 621 format |
| Modify | `setup.sh` | Replace Poetry with uv, add OS detection, remove sudo |
| Modify | `src/model/descriptor.py:175` | Fix Lightning 2.x dataloader access |
| Modify | `src/model/descriptor.py:206` | Pass coordinates to CurriculumTripletLoss |
| Modify | `src/model/losses.py:132` | Fix division-by-zero in spatial weights |
| Modify | `src/data/datamodule.py:280` | Fix return type annotation |
| Modify | `src/data/transforms.py:1-9` | Remove duplicate imports |
| Modify | `src/utils/sift.py:58,123,155,193` | Remove sudo, fix path, fix off-by-one |
| Modify | `scripts/create_heatmaps.py:131-132` | Add bounds check for empty mr/ dir |
| Modify | `example_test.py:103` | Fix patch_size config access |
| Modify | `README.md` | Add IEEE TMI badge, update citation, uv instructions |
| Modify | `.gitattributes` | Create for Git LFS tracking |
| Delete | `poetry.lock` | No longer needed after uv migration |

---

### Task 1: Switch from Poetry to uv and fix dependencies

**Files:**
- Modify: `pyproject.toml`
- Delete: `poetry.lock`

- [ ] **Step 1: Rewrite pyproject.toml for uv**

Remove `[tool.poetry]` and `poetry-core` build-system. Add missing `tqdm` and `scipy` dependencies. Fix PEP 508 dependency format (remove Poetry-style version constraints with parentheses).

```toml
[project]
name = "crosskey"
version = "0.1.0"
description = "A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration"
authors = [
    {name = "D. Morozov"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.7.1,<3.0.0",
    "torchvision>=0.22.1,<0.23.0",
    "torchaudio>=2.7.1,<3.0.0",
    "lightning>=2.5.2,<3.0.0",
    "pandas>=2.3.1,<3.0.0",
    "scikit-learn>=1.7.1,<2.0.0",
    "nibabel>=5.3.2,<6.0.0",
    "wandb>=0.21.3,<0.22.0",
    "tensorboard>=2.20.0,<3.0.0",
    "matplotlib>=3.10.6,<4.0.0",
    "tqdm>=4.66.0",
    "scipy>=1.14.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Remove poetry.lock**

```bash
rm poetry.lock
```

- [ ] **Step 3: Verify uv can resolve dependencies**

```bash
cd /path/to/CrossKEY
uv venv .venv
uv pip install -e .
```

Expected: All dependencies install without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore
git rm poetry.lock
git commit -m "chore: switch from Poetry to uv, add missing tqdm/scipy deps"
```

---

### Task 2: Fix critical code bugs

**Files:**
- Modify: `src/model/descriptor.py:175`
- Modify: `src/data/datamodule.py:280`
- Modify: `src/data/transforms.py:1-9`
- Modify: `example_test.py:103`

- [ ] **Step 1: Fix Lightning 2.x dataloader access in descriptor.py**

At line 175, `self.trainer.train_dataloader.dataset` is broken in Lightning 2.x. Replace with datamodule access:

```python
# Old (line 175):
dataset = self.trainer.train_dataloader.dataset

# New:
dataset = self.trainer.datamodule.train_dataset
```

- [ ] **Step 2: Fix example_test.py patch_size config access**

At line 103, `patch_size` is read from wrong config level. Fix to match `example_train.py` pattern:

```python
# Old (line 103):
patch_size=(config.get('patch_size', 32),) * 3,

# New:
patch_size=(config.get('data', {}).get('patch_size', 32),) * 3,
```

- [ ] **Step 3: Fix _load_heatmap_data return type annotation**

At line 280 of `datamodule.py`, the type annotation says 2 returns but function returns 3:

```python
# Old (line 280):
def _load_heatmap_data(self, heatmap_file: Path) -> Tuple[torch.Tensor, torch.Tensor]:

# New:
def _load_heatmap_data(self, heatmap_file: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

- [ ] **Step 4: Remove duplicate imports in transforms.py**

Lines 1-9 have duplicate imports. Remove lines 6-9:

```python
# Keep only these (lines 1-4):
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# Remove the duplicate block (lines 6-9):
# import torch
# import numpy as np
# import torch.nn.functional as F
# from typing import Optional, Tuple, Dict
```

Also need to keep `Dict` import since `BatchRotate3D.rotate_batch` uses `Dict`:

```python
from typing import Tuple, Optional, Dict
```

- [ ] **Step 5: Commit**

```bash
git add src/model/descriptor.py src/data/datamodule.py src/data/transforms.py example_test.py
git commit -m "fix: critical bugs - Lightning 2.x compat, config parsing, type annotations, duplicate imports"
```

---

### Task 3: Fix SIFT3D wrapper issues

**Files:**
- Modify: `src/utils/sift.py:58,123,155,193`

- [ ] **Step 1: Fix executable path to use project-root-relative resolution**

At line 58, replace the hardcoded relative path with a path resolved from the project root:

```python
# Old (line 58):
self.executable = './external_libs/SIFT3D/build/bin/kpSift3D'

# New:
self.executable = str(Path(__file__).resolve().parent.parent.parent / 'external_libs' / 'SIFT3D' / 'build' / 'bin' / 'kpSift3D')
```

Add `from pathlib import Path` to imports (already has `import os`).

- [ ] **Step 2: Remove sudo from run_sift3d command**

At line 123, remove `'sudo'` from the command list:

```python
# Old (lines 123-133):
command = ['sudo',
            self.executable,
           ...

# New (remove 'sudo'):
command = [self.executable,
           ...
```

- [ ] **Step 3: Remove sudo from get_sift_descriptors**

At line 193-194, fix the standalone function:

```python
# Old:
executable = './external_libs/SIFT3D/build/bin/featSift3D'
command = ['sudo',
            executable,

# New:
executable = str(Path(__file__).resolve().parent.parent.parent / 'external_libs' / 'SIFT3D' / 'build' / 'bin' / 'featSift3D')
command = [executable,
```

- [ ] **Step 4: Fix off-by-one in count_keypoints**

At lines 155-158, the CSV has a header row that gets counted:

```python
# Old:
def count_keypoints(self, keypoints_file):
    with open(keypoints_file, 'r') as f:
        csv_reader = csv.reader(f)
        return sum(1 for row in csv_reader)

# New:
def count_keypoints(self, keypoints_file):
    with open(keypoints_file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)  # Skip header row
        return sum(1 for row in csv_reader)
```

- [ ] **Step 5: Commit**

```bash
git add src/utils/sift.py
git commit -m "fix: SIFT3D wrapper - remove sudo, use project-relative paths, fix keypoint count off-by-one"
```

---

### Task 4: Fix important code issues

**Files:**
- Modify: `src/model/losses.py:132`
- Modify: `src/model/descriptor.py:206`
- Modify: `scripts/create_heatmaps.py:131-132`

- [ ] **Step 1: Fix division-by-zero in _compute_spatial_weights**

At line 132 of `losses.py`, `spatial_dist.max()` can be 0:

```python
# Old (line 132):
weights = spatial_dist / spatial_dist.max()

# New:
weights = spatial_dist / (spatial_dist.max() + 1e-8)
```

- [ ] **Step 2: Pass coordinates to CurriculumTripletLoss in training_step**

At line 206 of `descriptor.py`, the batch contains `point` coordinates but they're never passed to the loss. This disables spatial-aware negative mining:

```python
# Old (line 206):
loss, components = self.loss_fn(anchor_output, positive_output)

# New:
coordinates = batch.get('point', None)
loss, components = self.loss_fn(anchor_output, positive_output, coordinates=coordinates)
```

- [ ] **Step 3: Add bounds check in create_heatmaps.py**

At lines 131-132, `mr_files[0]` crashes if directory is empty:

```python
# Old (lines 131-132):
mr_files = list((self.data_dir / "mr").glob("*.nii.gz"))
self.reference_nifti_path = str(mr_files[0])

# New:
mr_files = list((self.data_dir / "mr").glob("*.nii.gz"))
if not mr_files:
    raise FileNotFoundError(f"No MR files found in {self.data_dir / 'mr'}. Place .nii.gz files there first.")
self.reference_nifti_path = str(mr_files[0])
```

- [ ] **Step 4: Commit**

```bash
git add src/model/losses.py src/model/descriptor.py scripts/create_heatmaps.py
git commit -m "fix: spatial weight NaN, enable coordinate-based mining, heatmap bounds check"
```

---

### Task 5: Update setup.sh for uv with OS detection

**Files:**
- Modify: `setup.sh`

- [ ] **Step 1: Rewrite setup.sh**

Replace Poetry with uv. Add OS detection for SIFT3D (Linux-only). Remove sudo from Python sections.

```bash
#!/bin/bash

set -e  # Exit on any error

echo "CrossKEY - Setup"
echo "=================================================="

# Check OS for SIFT3D support
OS="$(uname -s)"
if [ "$OS" != "Linux" ]; then
    echo "WARNING: SIFT3D compilation requires Linux (apt-get dependencies)."
    echo "On $OS, the Python environment will be set up but SIFT3D must be installed manually."
    echo "See: https://github.com/bbrister/SIFT3D"
    SKIP_SIFT3D=true
else
    SKIP_SIFT3D=false
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "uv found"

# Setup Python virtual environment with uv
echo "Setting up Python virtual environment with uv..."
uv venv .venv
uv pip install -e .

echo "Python dependencies installed"

# Install external libraries (Linux only)
if [ "$SKIP_SIFT3D" = false ]; then
    echo "Installing external libraries..."
    mkdir -p external_libs
    cd external_libs

    # Install SIFT3D
    echo "Installing SIFT3D..."
    echo "Installing system dependencies for SIFT3D..."
    sudo apt-get update && sudo apt-get install -y \
        zlib1g-dev \
        liblapack-dev \
        libdcmtk-dev \
        libnifti-dev \
        libblas-dev \
        cmake \
        build-essential

    if [ ! -d "SIFT3D" ]; then
        echo "Cloning SIFT3D repository..."
        git clone https://github.com/morozovdd/SIFT3D.git
    fi

    cd SIFT3D
    if [ ! -d "build" ]; then
        mkdir build
    fi
    cd build
    echo "Building SIFT3D..."
    cmake ..
    make -j$(nproc)
    cd ../..
    cd ..

    echo "SIFT3D installed successfully"
else
    echo "Skipping SIFT3D installation (not on Linux)"
    echo "You will need to install SIFT3D manually for full functionality."
fi

# Setup data directories
echo "Setting up data directories..."
mkdir -p data/heatmap
mkdir -p data/sift_output/mr data/sift_output/synthetic_us
mkdir -p logs

echo "Data directories created"

echo ""
echo "Setup completed!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Start training (automatically generates heatmaps and SIFT descriptors):"
echo "   python example_train.py"
echo ""
echo "3. Or test the installation first:"
echo "   python example_test.py"
echo ""
echo "Note: The training script will automatically run data preprocessing"
echo "      (SIFT extraction and heatmap generation) if needed."
echo ""
echo "For more information, see README.md"
```

- [ ] **Step 2: Commit**

```bash
git add setup.sh
git commit -m "chore: switch setup.sh to uv, add OS detection for SIFT3D"
```

---

### Task 6: Update README with IEEE TMI publication and uv instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add IEEE TMI badge and link**

After the arXiv badge line (line 19), add the IEEE TMI badge:

```html
<a href="https://ieeexplore.ieee.org/document/11474556"><img src="https://img.shields.io/badge/IEEE-TMI-blue" alt='IEEE TMI'></a>
```

- [ ] **Step 2: Update citation BibTeX**

Replace the arXiv citation (lines 174-183) with the IEEE TMI journal citation:

```bibtex
@article{morozov2026crosskey,
  title={A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration},
  author={Morozov, Daniil and Dorent, Reuben and Haouchine, Nazim},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  doi={10.1109/TMI.2026.3680352},
  url={https://ieeexplore.ieee.org/document/11474556}
}
```

- [ ] **Step 3: Update installation instructions for uv**

Replace Poetry references in the Quick Start section with uv:

Prerequisites change: `Poetry for dependency management` -> `uv for dependency management`

Installation step 2 stays as `./setup.sh`.

Step 3 changes from:
```bash
poetry shell
python example_train.py
```
to:
```bash
source .venv/bin/activate
python example_train.py
```

Usage section: replace `poetry run python` with `python` (assumes venv is active).

- [ ] **Step 4: Add shallow clone tip**

After the `git clone` line in Installation, add:

```bash
# For faster cloning (recommended):
git clone --depth 1 https://github.com/morozovdd/CrossKEY.git
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add IEEE TMI publication, update to uv, add shallow clone tip"
```

---

### Task 7: Git LFS + history rewrite to fix clone time

**Important:** This task requires force-pushing and will break existing clones. Do this last, after all other changes are committed and pushed.

**Files:**
- Create: `.gitattributes`
- Modify: `.gitignore`

- [ ] **Step 1: Install git-lfs if not already installed**

```bash
git lfs install
```

- [ ] **Step 2: Create .gitattributes for LFS tracking**

```
# Track large binary files with Git LFS
*.gif filter=lfs diff=lfs merge=lfs -text
*.nii.gz filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 3: Use git-filter-repo to purge old blobs from history**

First, make a backup of the repo:

```bash
cp -r /path/to/CrossKEY /path/to/CrossKEY-backup
```

Then run filter-repo to remove the old large files that were deleted but remain in history:

```bash
pip install git-filter-repo

git filter-repo --invert-paths \
  --path assets/mr_us_fov_comparison.gif \
  --path assets/mr_us_volume_matches_x.gif \
  --path assets/mr_us_volume_matches_y.gif \
  --path assets/mr_us_volume_matches_z.gif
```

- [ ] **Step 4: Migrate existing tracked large files to LFS**

```bash
git lfs migrate import --include="*.gif,*.nii.gz,*.png" --everything
```

- [ ] **Step 5: Add .gitattributes and verify repo size**

```bash
git add .gitattributes
git commit -m "chore: add Git LFS tracking for large binary files"
```

Verify the new repo size:

```bash
git count-objects -vH
```

Expected: size-pack should be significantly smaller (< 30 MB for non-LFS objects).

- [ ] **Step 6: Re-add remote and force-push**

After filter-repo, the remote is removed. Re-add it and force push:

```bash
git remote add origin https://github.com/morozovdd/CrossKEY.git
git push --force --all origin
git push --force --tags origin
```

- [ ] **Step 7: Verify clone time**

```bash
time git clone https://github.com/morozovdd/CrossKEY.git /tmp/crosskey-test
```

Expected: Clone time should be under 1 minute.

---

### Task 8: Respond to HuggingFace issue #1

- [ ] **Step 1: Submit paper to hf.co/papers**

Navigate to https://huggingface.co/papers/submit and submit arXiv paper 2507.18551.

- [ ] **Step 2: Post response on GitHub issue**

Post a reply to https://github.com/morozovdd/CrossKEY/issues/1 :

```
Hi @NielsRogge,

Thanks for reaching out! I've submitted the paper to hf.co/papers.

Regarding model weights: CrossKEY uses a patient-specific training approach (each model is trained on a specific patient's synthetic data), so releasing a single pretrained model wouldn't be broadly applicable. However, I'd love to explore building a demo on Spaces that showcases the matching pipeline - the ZeroGPU grant would be great for that.

I'll follow up once I have a demo ready. Thanks for the guidance on hosting and discoverability!
```

- [ ] **Step 3: Update README TODO list**

Update the TODO section to reflect current state:

```markdown
## TODO

- [X] **Essential Scripts**: Add training and testing scripts with test data example
- [ ] **Interactive Demo**: Create HuggingFace Spaces demo ([#1](https://github.com/morozovdd/CrossKEY/issues/1))
- [ ] **Visualization Functions**: Add utilities for keypoint and matching visualization
```
