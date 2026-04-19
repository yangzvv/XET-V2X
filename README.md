# XET-V2X: End-to-End 3D Spatiotemporal Perception with Multimodal Fusion and V2X Collaboration

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1%2BCUDA11.8-red.svg)](https://pytorch.org/)
[![MMCV 1.6.0](https://img.shields.io/badge/MMCV_Full-1.6.0-green.svg)](https://mmcv.readthedocs.io/)

## Highlights

**XET-V2X** is a multimodal fused end-to-end 3D spatiotemporal perception framework for vehicle-to-everything (V2X) collaboration.  
It unifies multiview multimodal sensing within a shared spatiotemporal representation, enabling robust detection and tracking under occlusions, limited viewpoints, and communication delays in cooperative driving scenarios.

This implementation provides:

- 🚀 End-to-end 3D detection and tracking with unified spatiotemporal modeling
- 🔄 Multiview and multimodal fusion for image–LiDAR cooperative perception
- 🌐 V2X collaboration with communication-delay-aware evaluation
- 📦 Compatibility with V2X-Seq-SPD, V2X-Sim-V2V, and V2X-Sim-V2I datasets
- 🛠️ Extensible architecture for multimodal cooperative perception research

---

## Getting Started

- **[Installation](docs/INSTALLATION.md)**: Detailed environment setup and dependency compilation.
- **[Data Preparation](docs/DATA_PREPARATION.md)**: Guide to download and process V2X-Seq, DAIR-V2X, and V2X-Sim datasets.
- **[Training and Evaluation](docs/TRAIN_EVAL.md)**: Step-by-step instructions to train, evaluate, and benchmark XET-V2X models across different cooperative perception scenarios.

## Models

For quick reproduction and performance evaluation, we provide our fully trained network checkpoints across multiple datasets. 

**Naming Convention Guide:**
* **Modality:** **`C`** stands for Camera-only, **`L`** stands for LiDAR-only, and **`X`** stands for Camera-LiDAR Fusion.
* **Perception Scope:** **`-V`** denotes single-agent perception (vehicle-side only), while **`-V2X`** denotes multi-agent cooperative perception.

| Model Name | Modality | Perception Scope | V2X-Seq-SPD | V2X-Sim-V2V | V2X-Sim-V2I |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CET-V** | Camera | Single-Agent | [ckpt](https://drive.google.com/file/d/1OHYw1GDVo3Aa6wlRpsHAW39vR6OVcZt5/view?usp=drive_link) \| [config](../projects/configs/v2x_seq_spd/cet_v(v2x).py) | [ckpt](https://drive.google.com/file/d/1XGpVBba_cQAOX2rBZNBwJ47cNaPbNyW5/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2v/cet_v(v2x).py) | [ckpt](https://drive.google.com/file/d/14umaQnFpyyli1eG0gsOf4u1ZT-lWcqcz/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2i/cet_v(v2x).py) |
| **LET-V** | LiDAR | Single-Agent | [ckpt](https://drive.google.com/file/d/1SHL6SBrFrUSBho8OGcuNatYIatN2_czt/view?usp=drive_link) \| [config](../projects/configs/v2x_seq_spd/let_v(v2x).py) | [ckpt](https://drive.google.com/file/d/1WaRL1-_N7pC3JzfXo2vCIE6Iq2gRbikK/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2v/let_v(v2x).py) | [ckpt](https://drive.google.com/file/d/1T4kABuMfcbYkaAa8nz-uB4uJfsTZd4i2/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2i/let_v(v2x).py) |
| **XET-V** | Camera + LiDAR | Single-Agent | [ckpt](https://drive.google.com/file/d/1bh60iBILTGl6jCkBwSTi8xmyXBxCwBui/view?usp=drive_link) \| [config](../projects/configs/v2x_seq_spd/xet_v(v2x).py) | [ckpt](https://drive.google.com/file/d/1YZw1Q_wD26eDzzDnSkW1TilvoSa1tKJf/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2v/xet_v(v2x).py) | [ckpt](https://drive.google.com/file/d/1y4PrN3YfwE1tj43C6bsBuHO1WbhYAZ8u/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2i/xet_v(v2x).py) |
| **CET-V2X** | Camera | Multi-Agent | [ckpt](https://drive.google.com/file/d/1F7nOt82Sfb_zZPeT3zg7RXDeZyJIqmWh/view?usp=drive_link) \| [config](../projects/configs/v2x_seq_spd/cet_v2x.py) | [ckpt](https://drive.google.com/file/d/1ggWbTxUNyu2rH8fX_Kq2HCozdIrRjfki/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2v/cet_v2x.py) | [ckpt](https://drive.google.com/file/d/1Dil_97DO5WFwSd-bX2p9txQd-97Xq-vh/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2i/cet_v2x.py) |
| **LET-V2X** | LiDAR | Multi-Agent | [ckpt](https://drive.google.com/file/d/17OXa8F-UnO-Tk0nOmmDJzZUFtd_H7IlN/view?usp=drive_link) \| [config](../projects/configs/v2x_seq_spd/let_v2x.py) | [ckpt](https://drive.google.com/file/d/1eqYsrLy34ezxW2CpLEAXCfm4r2n5eMeG/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2v/let_v2x.py) | [ckpt](https://drive.google.com/file/d/1qN7kxHh5_Yzd1wCtheFd0n_ISxyPO8b6/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2i/let_v2x.py) |
| **XET-V2X** | Camera + LiDAR | Multi-Agent | [ckpt](https://drive.google.com/file/d/1ieTanHBwkfy7xegl--hQHDsCYeyKcQCF/view?usp=drive_link) \| [config](../projects/configs/v2x_seq_spd/xet_v2x.py) | [ckpt](https://drive.google.com/file/d/1ZXpCzgqPEJxScsvuas4BofVK7rQ7cB2-/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2v/xet_v2x.py) | [ckpt](https://drive.google.com/file/d/1lEjgdkQp7kLUBBArmL5hTEIZbmKBc8Vb/view?usp=drive_link) \| [config](../projects/configs/v2x_sim_v2i/xet_v2x.py) |

### Expected Performance

Below is the performance comparison under different communication delay conditions across the three V2X benchmarks. Latency is measured in frame-level delay (1 frame = 100 ms for V2X-Seq-SPD, and 200 ms for both V2X-Sim datasets). 

**Evaluation Metrics:**
* **mAP & AMOTA**: Higher is better.
* **AMOTP**: Lower is better.

| Model | Latency | V2X-Seq-SPD <br> `mAP / AMOTA / AMOTP` | V2X-Sim-V2V <br> `mAP / AMOTA / AMOTP` | V2X-Sim-V2I <br> `mAP / AMOTA / AMOTP` |
| :--- | :---: | :---: | :---: | :---: |
| CET-V | - | 0.198 / 0.241 / 1.618 | 0.217 / 0.190 / 1.646 | 0.177 / 0.136 / 1.726 |
| LET-V | - | 0.449 / 0.437 / 1.175 | 0.424 / 0.315 / 1.401 | 0.379 / 0.308 / 1.432 |
| XET-V | - | 0.490 / 0.469 / 1.122 | 0.510 / 0.459 / 1.168 | 0.412 / 0.364 / 1.335 |
| CET-V2X | 0 | 0.179 / 0.169 / 1.630 | 0.313 / 0.291 / 1.419 | 0.714 / 0.782 / 0.567 |
| CET-V2X | 1 | 0.188 / 0.169 / 1.631 | 0.299 / 0.272 / 1.465 | 0.638 / 0.723 / 0.706 |
| CET-V2X | 2 | 0.187 / 0.169 / 1.633 | 0.278 / 0.240 / 1.499 | 0.572 / 0.626 / 0.826 |
| LET-V2X | 0 | 0.668 / 0.616 / 0.859 | 0.502 / 0.391 / 0.865 | 0.653 / 0.562 / 0.671 |
| LET-V2X | 1 | 0.608 / 0.592 / 0.951 | 0.462 / 0.373 / 0.898 | 0.573 / 0.511 / 0.817 |
| LET-V2X | 2 | 0.546 / 0.543 / 1.009 | 0.451 / 0.346 / 0.938 | 0.508 / 0.430 / 0.901 |
| **XET-V2X** | 0 | **0.795** / **0.787** / **0.591** | **0.766** / **0.731** / **0.677** | **0.858** / **0.819** / **0.476** |
| **XET-V2X** | 1 | **0.743** / **0.761** / **0.668** | **0.714** / **0.698** / **0.760** | **0.746** / **0.776** / **0.584** |
| **XET-V2X** | 2 | **0.688** / **0.722** / **0.730** | **0.668** / **0.652** / **0.804** | **0.664** / **0.668** / **0.712** |

## License

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

---
## Acknowledgements

This project is developed based on several excellent open-source codebases. We would like to express our gratitude to the authors of the following projects:

- [LET-VIC](https://github.com/yangzvv/LET-VIC): Our prior work that laid the groundwork for this research.
- [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq): For providing the comprehensive sequential dataset that drives vehicle-infrastructure cooperative research.
- [V2X-Sim](https://github.com/ai4ce/V2X-Sim): For the extensive simulated dataset and benchmark that greatly facilitated our V2X collaboration evaluation.
- [UniV2X](https://github.com/AIR-THU/UniV2X): For the foundational framework in cooperative V2X perception and modeling.
- [UniAD](https://github.com/OpenDriveLab/UniAD): For the pioneering work in end-to-end autonomous driving perception.
- [Argoverse API](https://github.com/argoverse/argoverse-api): For the essential tools provided for spatiotemporal data processing.

We also thank the researchers and developers in the V2X community for their invaluable contributions and for providing open-access datasets.