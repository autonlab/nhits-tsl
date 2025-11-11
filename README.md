# 🌀 N-HiTS for Time-Series-Library

> **Time Series Library for N-HiTS (AutonLab Edition)**  
> Independent implementation of the [N-HiTS paper (Challu et al., 2022)](https://arxiv.org/abs/2201.12886),  
> built on top of the [Time-Series-Library (TSL)](https://github.com/thuml/Time-Series-Library).

---

## 🚀 Overview

This repository provides an **implementation of N-HiTS** for time series forecasting  
within the TSL framework, adapted for standalone research and experiments.

---

## ⚙️ Installation

Clone this repository and install required dependencies.

```bash
git clone https://github.com/autonlab/nhits-tsl.git
cd nhits-tsl

# (optional) create a virtual environment

source .venv/bin/activate

# Additional packages required for N-HiTS experiments
uv pip install patool sktime PyWavelets reformer_pytorch
