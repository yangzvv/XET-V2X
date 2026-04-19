# Installation

## Requirements

The codebase has been validated under the following system environment:
- **OS:** Ubuntu 20.04/22.04
- **Python:** 3.9
- **PyTorch:** 2.0.1 (CUDA 11.8)
- **OpenMMLab:** MMCV-full 1.6.0, MMDetection 2.26.0, MMSegmentation 0.29.1, MMDetection3D 1.0.0rc6

> **Note:** We strongly recommend using a fresh Conda environment to avoid dependency conflicts.

## 1. Create and Activate Conda Environment

```bash
conda create -n xet-v2x python=3.9 -y
conda activate xet-v2x
```

## 2. Install System Dependencies (GCC & CUDA)

Compiling certain 3D operators requires modern C++ compilers. This project is built and tested using **GCC 9.4.0**. If your system's default GCC is outdated or differs, you can install the specific version via Conda:

```bash
conda install -c conda-forge gcc_linux-64=9.4.0 gxx_linux-64=9.4.0 -y
```

Ensure your ``CUDA_HOME`` is correctly set for compiling custom operators:
```bash
# Adjust the path below if your CUDA is installed elsewhere
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:$PATH
```

## 3. Install PyTorch

Install PyTorch, TorchVision, and TorchAudio with CUDA 11.8 support:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## 4. Clone the Repository

```bash
git clone https://github.com/yangzvv/XET-V2X.git
cd XET-V2X
```

## 5. Install OpenMMLab Ecosystem

We recommend using OpenMMLab's ``mim`` tool to streamline the installation of ``mmcv-full`` and other dependencies:

```bash
pip install -U openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1
mim install mmdet3d==1.0.0rc6
```

## 6. Install Project Requirements

Install the remaining Python dependencies required by XET-V2X:

```bash
pip install -r requirements.txt
```

## 7. Install Argoverse API

Certain components and data loaders depend on the Argoverse API. Install it from the source:

```bash
git clone https://github.com/argoverse/argoverse-api.git
cd argoverse-api
pip install -e .
cd ..
```