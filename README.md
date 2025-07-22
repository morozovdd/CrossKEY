<p align="center">
  <h1 align="center">🔑 <ins>CrossKEY</ins><br>A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration</h1>
  <p align="center">
    <a href="https://author1-website.com">Daniil Morozov</a><sup>1,2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=xdECLMkAAAAJ&hl=fr">Reuben Dorent</a><sup>3,4</sup>
    ·
    <a href="https://author3-website.com">Nazim Haouchine</a><sup>2</sup>
  </p>
  <p align="center">
    <sup>1</sup> Technical University of Munich (TUM), &nbsp;<sup>2</sup> Harvard Medical School, &nbsp;
    <sup>3</sup> Inria Saclay, &nbsp;
    <sup>4</sup> Sorbonne Université, Paris Brain Institute (ICM)
  </p>
</p>

<div align="center">

<a href="https://arxiv.org/abs/placeholder"><img src="https://img.shields.io/badge/arXiv-placeholder-b31b1b" alt='arxiv'></a>
<a href="https://colab.research.google.com/placeholder"><img src="https://img.shields.io/badge/Colab-Demo-F9AB00?logo=googlecolab&logoColor=white" alt='Colab Demo'></a>
<a href="https://www.cancerimagingarchive.net/collection/remind/"><img src="https://img.shields.io/badge/Dataset-ReMIND-blue" alt='Dataset'></a>
<a href="#license"><img src="https://img.shields.io/badge/License-MIT-green" alt='License'></a>

</div>

<div align="center">

<p align="center">
    <img src="assets/main.gif" alt="CrossKEY Demo" width="80%">
    <br>
    <em>CrossKEY enables robust 3D keypoint matching between MRI and iUS, achieving state-of-the-art performance both in image matching and registration tasks</em>
</p>

</div>

## 📋 Abstract

Intraoperative registration of real-time ultrasound (iUS) to preoperative Magnetic Resonance Imaging (MRI) remains an unsolved problem due to severe modality-specific differences in appearance, resolution, and field-of-view. To address this, we propose a novel 3D cross-modal keypoint descriptor for MRI–iUS matching and registration. Our approach employs a **patient-specific matching-by-synthesis approach**, generating synthetic iUS volumes from preoperative MRI. This enables supervised contrastive training to learn a shared descriptor space. A **probabilistic keypoint detection strategy** is then employed to identify anatomically salient and modality-consistent locations. During training, a curriculum-based triplet loss with dynamic hard negative mining is used to learn descriptors that are i) robust to iUS artifacts such as speckle noise and limited coverage, and ii) rotation-invariant. At inference, the method detects keypoints in MR and real iUS images and identifies sparse matches, which are then used to perform rigid registration. Our approach is evaluated using 3D MRI-iUS pairs from the ReMIND dataset. Experiments show that our approach outperforms state-of-the-art keypoint matching methods across 11 patients, with an average precision of **69.8%**. For image registration, our method achieves a competitive mean Target Registration Error of **2.39 mm** on the ReMIND2Reg benchmark.

<p align="center">
  <img src="assets/Overview.png" alt="Method Overview" width="100%"> 
</p>
<p align="center">
  <em>Overview of our CrossKEY framework</em>
</p>
