# 🩺 Pneumonia Detection from Chest X-Rays using CNN (TensorFlow)

This repository contains a **Convolutional Neural Network (CNN)** model built with **TensorFlow** for detecting **Pneumonia** from **chest X-ray images**. The model is designed with a strong emphasis on **high sensitivity**, ensuring that pneumonia cases are correctly identified as often as possible.

---

## 🚀 Project Overview

The goal of this project is to assist medical diagnosis by automatically classifying chest radiographs into:

- ✅ **Normal** (No Pneumonia)
- ✅ **Pneumonia** (Bacterial or Viral)

The CNN was trained and evaluated on a public dataset of chest X-rays, with special care taken to optimize the model for **high sensitivity** — i.e., minimizing false negatives in pneumonia detection.

---

## 🎯 Project Goals

- ✔️ Accurately detect **Pneumonia** from X-ray images
- ✔️ **Maximize sensitivity** to avoid missing positive cases
- ✔️ Use a clean and interpretable TensorFlow/Keras implementation

> 🧪 **Achieved Sensitivity (Pneumonia class): 95%**

---

## 🧠 Model Architecture

- Custom CNN built with TensorFlow/Keras
- Convolutional + MaxPooling layers
- Dropout regularization
- Fully connected classification head
- Binary classification (sigmoid activation)

> Optionally: Swap in a pre-trained model like **ResNet50** or **MobileNetV2** for better performance on small datasets.

---

## 🧬 Dataset

- Public chest X-ray dataset containing labeled images of:
  - Normal lungs
  - Pneumonia (viral or bacterial)

📦 Dataset source: [Kaggle Pneumonia Dataset] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

---

## 📊 Results

| Metric         | Score     |
|----------------|-----------|
| Accuracy       | ~91%      |
| **Sensitivity**| **97%** ✅|
| Specificity    | ~91%      |

> 📌 Sensitivity was the key target metric, as false negatives are critical in medical contexts.

---

### 🔧 Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib, Scikit-learn
- (Optional: OpenCV, Pillow for image preprocessing)





