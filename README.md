<div align="center">

# 🍑 Fruit Freshness Neural Network 🍓

**Feedforward Neural Network with Image Feature Extraction**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-image%20processing-orange)
![No DL Frameworks](https://img.shields.io/badge/TensorFlow%20%2F%20PyTorch%20%2F%20Keras-not%20used-red)

</div>

> 🎓 **Course:** Artificial Neural Network and Deep Learning
> 🧠 Feedforward Neural Network with Image Feature Extraction

---

## 📚 Table of Contents

1. [🎯 Project Overview](#-1-project-overview)
2. [🗂️ Dataset](#️-2-dataset)
3. [🧠 Architecture](#-3-architecture)
4. [🖼️ Image Feature Extraction](#️-4-image-feature-extraction)
5. [⚙️ Activation and Transfer Functions](#️-5-activation-and-transfer-functions)
6. [🏋️ Training Pipeline](#️-6-training-pipeline)
7. [🚀 How to Run](#-7-how-to-run)
8. [📋 Output Explained](#-8-output-explained)
9. [📦 Dependencies](#-9-dependencies)

---

## 🎯 1. Project Overview

This program implements a **feedforward neural network from scratch (using only NumPy)** to classify the freshness of two fruits, **Peaches** and **Strawberries**, using image processing.

### 💡 Key Ideas

- 🖼️ Images cannot be fed directly into a neural network as raw pixel data.
- 🔢 We extract compact **9-dimensional numerical feature vectors** per image.
- 🍑🍓 Two images (one peach, one strawberry) are passed together and their features are concatenated into a single **18-dimensional input vector**.
- ✅ The network predicts whether **BOTH** fruits are fresh simultaneously.

---

## 🗂️ 2. Dataset

### 📁 Dataset Structure

```text
Fruits Dataset/
├── fresh_peaches_done/        250 fresh peach images
├── rotten_peaches_done/       343 rotten peach images
├── fresh_strawberries_done/   250 fresh strawberry images
└── rotten_strawberries_done/  251 rotten strawberry images
```

### 📊 Dataset Summary

| Fruit | 🟢 Fresh | 🟤 Rotten | Total |
|:------|:-------:|:--------:|:-----:|
| 🍑 Peach | 250 | 343 | 593 |
| 🍓 Strawberry | 250 | 251 | 501 |

> 🔗 **Combined dataset:** We pair one peach image with one strawberry image per sample (**501 total pairs**). Label = `1` only when **BOTH** fruits are fresh (AND logic).

---

## 🧠 3. Architecture

### 🔗 Network Structure

```text
Input (18) → Hidden1 (16, ReLU) → Hidden2 (8, ReLU) → Hidden3 (4, ReLU) → Output (1, Sigmoid)
```

### 🧱 Layer Details

| Layer | Neurons | Activation |
|:------|:-------:|:----------:|
| Input | 18 | — |
| Hidden 1 | 16 | ReLU |
| Hidden 2 | 8 | ReLU |
| Hidden 3 | 4 | ReLU |
| Output | 1 | Sigmoid |

Every neuron in each layer is fully connected to every neuron in the next layer.

### ⚖️ Weight Initialisation

**Weight Initialisation:** He (Kaiming) initialisation `W ~ N(0, sqrt(2/fan_in))`

This is the optimal starting point for ReLU networks, preventing vanishing/exploding gradients.

---

## 🖼️ 4. Image Feature Extraction

Each image is resized to **64x64 pixels**, then a **9-element feature vector** is extracted:

| Index | Feature | Description |
|:-----:|:--------|:------------|
| 0 | 🔴 Mean Red | Average red channel value |
| 1 | 🟢 Mean Green | Average green channel value |
| 2 | 🔵 Mean Blue | Average blue channel value |
| 3 | 📈 Std Red | Spread of red channel (texture proxy) |
| 4 | 📈 Std Green | Spread of green channel |
| 5 | 📈 Std Blue | Spread of blue channel |
| 6 | 💡 Brightness | Overall luminance (mean greyscale) |
| 7 | 🔆 Contrast | Std of greyscale (sharpness indicator) |
| 8 | 🍎 Red-to-Green Ratio | Ripeness indicator (ripe fruits are redder) |

**Two fruits combined:** `[9 peach features] ++ [9 strawberry features] = 18-dim input`

---

## ⚙️ 5. Activation and Transfer Functions

### 🔹 ReLU (Hidden Layers)

```text
ReLU(z) = max(0, z)
```

- Used in all three hidden layers during forward pass and back-propagation.
- Prevents vanishing gradients; derivative is 0 or 1 (very efficient).

### 🔹 Sigmoid (Output Layer — Training)

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

- Output in (0, 1) interpreted as freshness probability.
- Used as primary output during training and evaluation.

### 🔹 Hard-Limit / Heaviside Step (Transfer Function 1)

```text
H(z) = 1 if z >= threshold, else 0
```

- Crisp binary output: fresh or not fresh.
- Cannot be used in training (not differentiable) but ideal for final decisions.

### 🔹 Sigmoid (Transfer Function 2)

Same sigmoid formula applied to raw logits for comparison display.

### 🔹 Tanh (Transfer Function 3)

```text
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

- Output in (-1, 1). Symmetric and zero-centred.
- `+1` = fresh, `-1` = rotten. Stronger gradients near 0 than sigmoid.

---

## 🏋️ 6. Training Pipeline

| Step | Description |
|:----:|:------------|
| 1️⃣ | Load all images from disk. |
| 2️⃣ | Resize to 64x64, extract 9 features per image. |
| 3️⃣ | Pair peach + strawberry features → 18-dim input; label = AND of labels. |
| 4️⃣ | Min-Max normalise all features to [0, 1]. |
| 5️⃣ | 80/20 train-test split. |
| 6️⃣ | Forward pass: X → ReLU → ReLU → ReLU → Sigmoid → y_hat. |
| 7️⃣ | Compute Binary Cross-Entropy loss. |
| 8️⃣ | Backward pass: compute gradients via chain rule. |
| 9️⃣ | Update weights: `W = W - lr * gradient` (lr = 0.01). |
| 🔟 | Repeat steps 6-9 for 200 epochs. |
| 1️⃣1️⃣ | Evaluate on test set, demonstrate transfer functions. |

### 📉 Loss Function: Binary Cross-Entropy

```text
L = -mean(y * log(y_hat) + (1-y) * log(1-y_hat))
```

Penalises confident wrong predictions heavily.

### 📈 Optimiser: Vanilla Gradient Descent (lr = 0.01)

The foundational algorithm; basis for all advanced optimisers.

---

## 🚀 7. How to Run

### 📦 Prerequisites (install once)

```bash
pip install numpy pillow
```

### ▶️ Run the Program

```bash
python main.py
```

> ⚠️ Make sure the `Fruits Dataset` folder is in the same directory as `main.py`.

---

## 📋 8. Output Explained

1. **📂 Loading Images & Extracting Features**
   Scans all peach/strawberry folders, extracts 9-feature vectors.

2. **🧠 Network Architecture**
   Prints the layer sizes (18 → 16 → 8 → 4 → 1) and settings.

3. **🏋️ Training (200 Epochs)**
   Prints loss every 20 epochs. Loss should decrease, showing the network learns.

4. **🧪 Evaluation on Test Set**
   Test loss and accuracy on the 20% held-out data.

5. **🔄 Transfer & Activation Function Demonstrations**
   Applies Hard-Limit, Sigmoid, and Tanh to raw output logits (Z4) side-by-side.

6. **🔍 Sample Predictions (First 10 Test Samples)**
   Table showing Sigmoid output, Hard-Limit decision, Tanh value, and true label for 10 test samples.

---

## 📦 9. Dependencies

| Package | Purpose | Install Command |
|:--------|:--------|:----------------|
| `numpy` | All matrix math / NN ops | `pip install numpy` |
| `Pillow` | Image loading and resizing | `pip install pillow` |

> 🚫 **No TensorFlow, PyTorch, or Keras used.**
> The entire neural network is implemented from mathematical first principles.
