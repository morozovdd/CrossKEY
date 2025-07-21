<p align="center">
  <h1 align="center">🔑 <ins>CrossKEY</ins><br>A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration</h1>
  <p align="center">
    <a href="https://author1-website.com">Author Name 1</a><sup>1</sup>
    ·
    <a href="https://author2-website.com">Author Name 2</a><sup>2†</sup>
    ·
    <a href="https://author3-website.com">Author Name 3</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup> Institution 1, Department &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup> Institution 2, Department<br>
    <sup>†</sup> Corresponding author
  </p>
  <h2 align="center">
    <p>Conference/Journal 2025</p>
    <a href="https://arxiv.org/abs/placeholder" align="center">📄 Paper</a> | 
    <a href="https://colab.research.google.com/placeholder" align="center">🚀 Demo</a> | 
    <a href="https://remind-dataset.github.io/" align="center">📊 Dataset</a> | 
    <a href="#citation" align="center">📝 Citation</a>
  </h2>
</p>

<div align="center">

<a href="https://arxiv.org/abs/placeholder"><img src="https://img.shields.io/badge/arXiv-placeholder-b31b1b" alt='arxiv'></a>
<a href="https://colab.research.google.com/placeholder"><img src="https://img.shields.io/badge/Colab-Demo-F9AB00?logo=googlecolab&logoColor=white" alt='Colab Demo'></a>
<a href="https://remind-dataset.github.io/"><img src="https://img.shields.io/badge/Dataset-ReMIND-blue" alt='Dataset'></a>
<a href="#license"><img src="https://img.shields.io/badge/License-MIT-green" alt='License'></a>

</div>

<p align="center">
    <img src="assets/mr_us_volume_matches_x.gif" alt="CrossKEY Demo" width="70%">
    <br>
    <em>CrossKEY enables robust 3D keypoint matching between MRI and ultrasound modalities,<br>achieving state-of-the-art performance with <strong>69.8% precision</strong> and <strong>2.39mm TRE</strong>.</em>
</p>

## 📋 Abstract

Intraoperative registration of real-time ultrasound (iUS) to preoperative Magnetic Resonance Imaging (MRI) remains an unsolved problem due to severe modality-specific differences in appearance, resolution, and field-of-view. To address this, we propose a novel 3D cross-modal keypoint descriptor for MRI–iUS matching and registration.

Our approach employs a **patient-specific matching-by-synthesis approach**, generating synthetic iUS volumes from preoperative MRI. This enables supervised contrastive training to learn a shared descriptor space. A **probabilistic keypoint detection strategy** is then employed to identify anatomically salient and modality-consistent locations. During training, a curriculum-based triplet loss with dynamic hard negative mining is used to learn descriptors that are i) robust to iUS artifacts such as speckle noise and limited coverage, and ii) rotation-invariant.

At inference, the method detects keypoints in MR and real iUS images and identifies sparse matches, which are then used to perform rigid registration. Our approach is evaluated using 3D MRI-iUS pairs from the ReMIND dataset. Experiments show that our approach outperforms state-of-the-art keypoint matching methods across 11 patients, with an average precision of **69.8%**. For image registration, our method achieves a competitive mean Target Registration Error of **2.39 mm** on the ReMIND2Reg benchmark.

<p align="center">
  <img src="assets/Method Overview.pdf" alt="Method Overview" height="300"> 
</p>
<p align="center">
  <em>Overview of our CrossKEY framework for cross-modal MRI-ultrasound matching and registration</em>
</p>

## 🎯 Key Features

- **🔄 Cross-modal Learning**: Novel 3D descriptor that bridges MRI and ultrasound modalities
- **👤 Patient-specific Approach**: Matching-by-synthesis strategy using synthetic iUS generation
- **🎯 Robust Keypoint Detection**: Probabilistic detection of anatomically salient locations
- **🚀 No Manual Initialization**: Fully automated registration pipeline
- **📐 Field-of-view Invariant**: Robust to varying ultrasound coverage
- **⚡ Real-time Performance**: Efficient inference for clinical workflows

---

## 📊 Results Overview

### 3D Cross-modal Volume Matching

Our method successfully identifies corresponding anatomical structures across MRI and ultrasound volumes in all three spatial dimensions:

<div align="center">

|                    X-axis Matches                    |                    Y-axis Matches                    |                    Z-axis Matches                    |
| :--------------------------------------------------: | :--------------------------------------------------: | :--------------------------------------------------: |
| ![X-axis matches](assets/mr_us_volume_matches_x.gif) | ![Y-axis matches](assets/mr_us_volume_matches_y.gif) | ![Z-axis matches](assets/mr_us_volume_matches_z.gif) |

_Cross-modal keypoint matches visualized across different anatomical planes_

</div>

### Field-of-View Invariance

CrossKEY demonstrates robust performance across varying ultrasound field-of-view conditions, maintaining accurate correspondence even with limited coverage:

<div align="center">

![FOV Comparison](assets/mr_us_fov_comparison.gif)

_Robust matching performance across different ultrasound field-of-view configurations_

</div>

### Registration Performance

The identified keypoint correspondences enable accurate rigid registration between preoperative MRI and intraoperative ultrasound:

<div align="center">

|            Registration Example 1            |            Registration Example 2            |
| :------------------------------------------: | :------------------------------------------: |
| ![Registration 1](assets/registration_1.gif) | ![Registration 2](assets/registration_2.gif) |

_Real-time MRI-ultrasound registration results showing alignment convergence_

</div>
