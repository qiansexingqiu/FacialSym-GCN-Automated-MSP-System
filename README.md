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

## 🏗️ Setup

### 1.Environment Setup
We recommend creating a clean conda environment:
```
conda create -n FacialSymGCN python=3.8 -y
conda activate FacialSymGCN
```

### 2.Required Dependencies
All dependencies used in our experiments are listed below (exact versions ensure reproducibility):
```
cuda == 11.3
torch == 1.11.0+cu113
torchvision == 0.12.0+cu113
scikit-learn == 1.0.2
pickleshare == 0.7.5
ninja == 1.10.2.3
SimpleITK == 2.3.0
gdown == 5.2.0
easydict == 1.9
PyYAML == 6.0
protobuf == 3.20.3
tensorboard == 2.8.0
termcolor == 1.1.0
tqdm == 4.62.3
multimethod == 1.7
h5py == 3.6.0
matplotlib == 3.5.1
wandb == 0.21.0
pyvista == 0.44.2
setuptools == 59.5.0
Cython == 0.29.28
pandas == 2.0.3
deepspeed == 0.17.2
shortuuid == 1.0.13
mkdocs-material == 9.6.16
mkdocs-awesome-pages-plugin == 2.10.1
mdx_truly_sane_lists == 1.3
vtk == 9.3.1
numpy == 1.24.4
numpy-stl == 3.1.2
acvl-utils == 0.2.1
dynamic-network-architectures == 0.4.2
scipy == 1.10.1
batchgenerators == 0.25.1
scikit-image == 0.21.0
graphviz == 0.20.3
tifffile == 2023.7.10
requests == 2.32.4
nibabel == 5.2.1
seaborn == 0.13.2
imagecodecs == 2023.3.16
yacs == 0.1.8
einops == 0.8.1
blosc2 == 2.2.2
trimesh == 4.7.1
nnunet_v2 == 2.0
```
### 3. Compile Required CUDA Extensions
(1) OpenPoints framework
```
cd OpenPoints_framework
source update.sh
```
(2) PointNet++ and related CUDA ops
```
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

cd subsampling
python setup.py build_ext --inplace
cd ..

cd pointops/
python setup.py install
cd ..

cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../
```
These steps compile all CUDA-based point cloud operators required for Stage II (DeepGCN-based bilateral segmentation).
### 4.Patch for acvl-utils (Bounding Box Cropping)
Due to a missing utility in the original acvl-utils package, one additional function must be manually added.
Edit the following file in your conda environment:
```
/home/[your_id]/anaconda3/envs/[your_env]/lib/python3.8/site-packages/acvl_utils/cropping_and_padding/bounding_boxes.py
```
Add this function:
```
def crop_to_bbox(data: np.ndarray, bbox: tuple):
    assert len(bbox) % 2 == 0,
    ndim = len(bbox) // 2
    slicer = []
    for d in range(ndim):
        start = int(bbox[2 * d])
        end = int(bbox[2 * d + 1])
        slicer.append(slice(start, end))
    return data[tuple(slicer)]
```

---

## 🚀 start inference MSP
If you want to inference MSP direct from CT, please run follow code:
```
python /function/from_stage1/run_msp.py
```
If you want to inference MSP starting from stage 2, please run follow code:
```
python /function/from_stage2/run_msp.py
```

