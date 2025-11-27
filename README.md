
# SatQuant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/](https://colab.research.google.com/github/gulis-dev/SatQuant/blob/main/assets/notebook.ipynb))
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/oskar-andrukiewicz/)

**Domain-Adaptive Post-Training Quantization (PTQ) for Satellite & Aerial Imagery.**

SatQuant is a Python framework designed to optimize Deep Learning models (YOLO, MobileNet, EfficientDet) for deployment on edge devices (Edge TPU, ARM, DSP). It addresses the specific challenge of small object degradation during INT8 quantization in high-resolution aerial data.

![SatQuant Benchmark Results](assets/benchmark.png)

## Overview

Standard quantization algorithms (e.g., TFLite default calibration) optimize parameters to minimize global reconstruction error (MSE). In satellite imagery, where >95% of pixels represent background (ocean, land), the quantizer allocates the dynamic range to the background. Consequently, small objects (ships, vehicles) are compressed into a single quantization bin and vanish.

**SatQuant solves this via Focus Calibration:** A strategy that shifts the calibration data distribution towards regions of interest, forcing the quantization engine to preserve high-frequency features of small objects.

## Key Features

- **Distribution Shift:** Alters calibration histograms to favor object features over background noise.
- **Model Agnostic:** Works with **TensorFlow SavedModel** (YOLOv8) and **Keras** (`.h5`, `.keras`) formats.
- **Normalization Control:** Optional flag to toggle input normalization (0-1 vs 0-255) depending on your model's architecture.
- **Artifact-Free Processing:** Implements a resizing strategy that prevents zero-padding artifacts from corrupting the quantization `ZeroPoint`.
- **Hardware-Aware:**
    - `full_int8`: Enforces strict INT8 I/O for Google Coral Edge TPU / NPU.
    - `mixed`: Hybrid FP32/INT8 execution for CPU-bound devices.

## Mathematical Principle

The core advantage of SatQuant lies in the minimization of the quantization Scale parameter ($S$).

Affine quantization maps real values ($r$) to integers ($q$):
$$r = S(q - Z)$$

The Scale ($S$) determines the granularity of the model:
$$S = \frac{r_{max} - r_{min}}{255}$$

### Comparison: Standard vs. SatQuant

| Metric | Standard Calibration | SatQuant (Focus Calibration) | Impact |
| :--- | :--- | :--- | :--- |
| **Dynamic Range** | Full Image (Background dominated) | Object Crops (Feature dominated) | Range reduced by ~75% |
| **Scale ($S$)** | High (e.g., ~0.06) | Low (e.g., ~0.015) | **4x Higher Precision** |
| **Bit Density** | Allocated to Background | Allocated to Objects | Critical features preserved |

By mathematically narrowing the dynamic range ($r_{max} - r_{min}$) to the object's histogram, SatQuant increases the resolution of the INT8 representation for critical targets.

## Performance Benchmark

**Target:** YOLOv8 Nano | **Task:** Small Vehicle Detection | **Input:** 640x640

| Pipeline | Precision | Model Size | Hardware Compatibility |
| :--- | :--- | :--- | :--- |
| **FP32 Baseline** | 100% (Ref) | 12.1 MB | GPU / CPU |
| **Standard TFLite** | 0% (Signal Loss) | 3.2 MB | Edge TPU / CPU |
| **SatQuant INT8** | **~76% (Recovered)** | **3.2 MB** | **Edge TPU / CPU** |

*(Note: "Standard TFLite" often results in 0 confidence for small objects due to aggressive dynamic range compression.)*

## Installation

**From Source (Recommended for Developers):**
```bash
git clone [https://github.com/gulis-dev/satquant.git](https://github.com/gulis-dev/satquant.git)
cd satquant
pip install -e .
````

**Direct Install (For Users):**

```bash
pip install git+[https://github.com/gulis-dev/satquant.git](https://github.com/gulis-dev/satquant.git)
```

## Usage

### 1\. Data Preparation

Ensure your dataset follows the DOTA format (images + `.txt` label files).

### 2\. Optimization Pipeline

```python
from satquant import FocusQuantizer, DotaDataset

# 1. Initialize Dataset with Context-Aware Cropping
# padding_pct=0.2 ensures the model learns local contrast (object vs. immediate background)
dataset = DotaDataset(
    images_dir="./dota_samples/images", 
    labels_dir="./dota_samples/labelTxt", 
    crop_size=640,
    padding_pct=0.2
)

# 2. Load Model (SavedModel or Keras .h5)
quantizer = FocusQuantizer(model_path="./yolov8n_saved_model")

# 3. Quantize
quantizer.convert(
    dataset=dataset,
    output_path="yolov8_sat_optimized.tflite",
    mode="full_int8",
    normalize_input=True  # Set False if your model expects raw 0-255 pixels
)
```

## Supported Model Formats

| Format | Extension | Notes |
| :--- | :--- | :--- |
| **SavedModel** | Directory | Standard export from YOLOv8 (`yolo export format=saved_model`) |
| **Keras** | `.h5`, `.keras` | Legacy TensorFlow / Custom Research Models |

## Project Structure

  - `satquant.data`: Handles OBB parsing, context padding, and crop generation.
  - `satquant.core`: Wraps TensorFlow Lite Converter with hardware-specific constraints.

## Disclaimer

This library is a tool for **Post-Training Quantization (PTQ)**. It assumes the baseline floating-point model is already capable of detecting objects in the target domain. SatQuant cannot fix a model that was not properly trained on satellite imagery.

## License

MIT License
