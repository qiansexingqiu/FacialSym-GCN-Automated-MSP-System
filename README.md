# FacialSym-GCN: Automated Midsagittal Plane Construction for Digital Surgical Planning  
**Open-source implementation of the MSP system described in:**  
Tao et al., *Real-world clinical performance of Automated Midsagittal Plane System for Digital Surgical Planning:  
Open-Source Implementation, Multicenter Evaluation, and Prospective Surgical Validation* (2025)

---

## 🧭 Overview

FacialSym-GCN is a fully automated deep learning framework for constructing the midsagittal plane (MSP) directly from computed tomography (CT) data.  
It integrates:

- **Anatomy-Guided Refocusing (AGR):** voxel-level segmentation of cranio-maxillofacial structures.
- **Segmentation-Guided Bilateral Fitting (SGBF):** point cloud graph convolution + geometric bilateral fitting.
- **Direct-from-CT inference:** no manual landmarks required.
- **Open-source reproducibility:** full code + weights + internal test set.

This repository provides the **official implementation**, trained weights, evaluation scripts, and the **open-source internal test dataset** used in the manuscript.

---

## 📌 Key Features

- ✔ **Fully automated MSP construction**
- ✔ **Direct-from-CT processing (no landmarks needed)**
- ✔ **nn-U-Net-based refocusing segmentation**
- ✔ **DeepGCN-based bilateral segmentation**
- ✔ **Surgery-level geometric accuracy**
- ✔ **Open-source internal test set for reproducible benchmarking**

---

## 🖼️ Graphical Abstract

![Graphical Abstract](./graphic_abstract.png)

---

## 🏗️ Repository Structure

---

## 🚀 Quick Start

