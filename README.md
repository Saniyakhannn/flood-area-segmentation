#  Flood Area Segmentation — UNet vs Attention UNet

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning project for flood area segmentation from aerial/drone imagery using UNet and Attention UNet architectures with Grad-CAM explainability.

---

##  Results

| Model | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| U-Net | 0.4393 | 0.5644 | 0.8426 | 0.4815 |
| **Attention U-Net** | **0.7756** | **0.8658** | **0.8690** | **0.8798** |

**Attention U-Net achieves 76.6% improvement in IoU over standard U-Net!**

---

## Project Structure
