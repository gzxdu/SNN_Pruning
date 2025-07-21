# Dynamic Spatio-Temporal Pruning for Efficient Spiking Neural Networks

This repository contains the implementation of **Dynamic Spatio-Temporal Pruning for Efficient Spiking Neural Networks** as described in the paper:  
**[Dynamic Spatio-Temporal Pruning for Efficient Spiking Neural Networks](https://doi.org/10.3389/fnins.2025.1545583)** (doi:10.3389/fnins.2025.1545583).

---

## Setup

Before using the code, install the required dependencies:

```bash
pip install pytorch
pip install spikingjelly
```

---

## Description

This repository provides training scripts for the **VGGSNN** network on the **CIFAR10-DVS dataset** with different pruning strategies:

- **`train_spiking_snn_cifar10dvs_VGGSNN.py`**: Implements **spatial pruning**.
- **`train_spiking_snn_cifar10dvs_VGGSNN_T.py`**: Implements **spatio-temporal pruning**.

Both scripts enable efficient training of spiking neural networks with reduced computational costs.

---

## Citation

If you find this repository or the associated paper useful, please cite:

```bibtex
@article{gou2025dynamic,
  title={Dynamic spatio-temporal pruning for efficient spiking neural networks},
  author={Gou, Shuiping and Fu, Jiahui and Sha, Yu and Cao, Zhen and Guo, Zhang and Eshraghian, Jason K and Li, Ruimin and Jiao, Licheng},
  journal={Frontiers in Neuroscience},
  volume={19},
  pages={1545583},
  year={2025},
  publisher={Frontiers Media SA}
}
