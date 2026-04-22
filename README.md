# AVO-denoising-with-AI
Pre conditioning data using machine learning
# 🌊 Amplitude-Preserving Seismic Denoising using U-Net

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Geophysics](https://img.shields.io/badge/Domain-Geophysics-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Seismic pre-conditioning is a critical step in reservoir characterization. However, traditional and aggressive denoising algorithms often destroy true amplitude anomalies, rendering the data unreliable for **Amplitude Versus Offset (AVO) analysis**. 

This repository contains an end-to-end Physics-Informed Machine Learning pipeline that automates seismic denoising while strictly preserving amplitude ratios and phase reversals across AVO Classes I, II, III, and IV.

By utilizing a **U-Net Convolutional Neural Network (CNN)** with skip connections, the model learns to separate random noise from geological reflectors without clipping or altering the foundational Aki-Richards physics.

## ✨ Key Features
* **Physics-Backed Synthetic Generation:** Automatically builds forward models using the Aki-Richards approximation and Ricker wavelets for all 4 AVO classes based on real-world petrophysical parameters ($V_p, V_s, \rho$).
* **Amplitude-Preserving AI:** A custom U-Net architecture optimized for regression tasks to retain spatial features and true amplitude gradients.
* **Rigorous Blind Testing:** Evaluates the model on completely unseen data (shifted frequencies and severe SNR levels) to prevent overfitting and prove geological generalization.
* **Industry Standard Export:** Capable of exporting generated synthetic gathers into `.sgy` (SEG-Y) format using `segyio` for downstream interpretation in software like Petrel or OpendTect.

---

## 🛠️ Tech Stack
* **Core:** Python
* **Deep Learning:** TensorFlow / Keras
* **Mathematics & Matrices:** NumPy
* **Visualization:** Matplotlib
* **Geophysics Format:** Segyio

---

## 🚀 Project Pipeline & Usage

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/seismic-unet-denoising.git](https://github.com/yourusername/seismic-unet-denoising.git)
cd seismic-unet-denoising
pip install numpy tensorflow matplotlib segyio
