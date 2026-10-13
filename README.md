<div align="center">
🍑 Fruit Freshness Neural Network
A NumPy-based feedforward neural network for peach and strawberry freshness classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Neural%20Network-013243?logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-yellow)
</div>
---
📚 Table of Contents
Overview
Dataset
Model Architecture
Image Feature Extraction
Activation and Transfer Functions
Training Pipeline
Getting Started
Understanding the Output
Dependencies
---
🎯 Overview
This project builds a feedforward neural network from scratch using NumPy to classify the combined freshness of peaches and strawberries.
Extracts 9 numerical features from each fruit image.
Combines peach and strawberry features into an 18-value input vector.
Predicts `1` only when both fruits are fresh; otherwise, predicts `0`.
> The model uses handcrafted image features rather than raw pixels and does not rely on TensorFlow, PyTorch, or Keras.
🗂️ Dataset
Folder Structure
```text
Fruits Dataset/
├── fresh\_peaches\_done/        # 250 images
├── rotten\_peaches\_done/       # 343 images
├── fresh\_strawberries\_done/   # 250 images
└── rotten\_strawberries\_done/  # 251 images
```
Dataset Summary
Fruit	Fresh	Rotten	Total
Peach	250	343	593
Strawberry	250	251	501
Combined samples: 501 peach–strawberry pairs. A sample is labeled `1` only when both fruits are fresh (AND logic).
🧠 Model Architecture
```text
Input (18) → Dense (16, ReLU) → Dense (8, ReLU)
           → Dense (4, ReLU) → Output (1, Sigmoid)
```
Layer	Neurons	Activation
Input	18	—
Hidden 1	16	ReLU
Hidden 2	8	ReLU
Hidden 3	4	ReLU
Output	1	Sigmoid
All adjacent layers are fully connected. Weights use He (Kaiming) initialization, suitable for ReLU layers.
🖼️ Image Feature Extraction
Each image is resized to 64 × 64 pixels and represented by 9 features.
Feature	Description
Mean Red, Green, Blue	Average value of each color channel
Std Red, Green, Blue	Variation in each color channel
Brightness	Mean grayscale intensity
Contrast	Standard deviation of grayscale intensity
Red-to-Green Ratio	Color ratio used as a ripeness-related indicator
Input construction: 9 peach features + 9 strawberry features = 18 features.
⚙️ Activation and Transfer Functions
Function	Purpose
ReLU — `max(0, z)`	Hidden-layer activation during training
Sigmoid — `1 / (1 + e⁻ᶻ)`	Converts the output logit into a probability between 0 and 1
Hard Limit / Heaviside	Converts a value into a binary decision using a threshold
Tanh	Maps values to the range −1 to 1 for comparison
The hard-limit function is useful for final binary decisions but is not differentiable, so it is not used for gradient-based training.
🏋️ Training Pipeline
Load images from the dataset folders.
Resize each image and extract its 9 features.
Pair peach and strawberry features; assign labels using AND logic.
Normalize features to the range `\[0, 1]`.
Split the data into 80% training and 20% testing.
Run forward propagation through the network.
Calculate Binary Cross-Entropy loss.
Backpropagate gradients and update weights using gradient descent.
Train for 200 epochs with a learning rate of `0.01`.
Evaluate the test set and compare transfer-function outputs.
Loss: Binary Cross-Entropy  
Optimizer: Vanilla Gradient Descent  
Learning rate: `0.01` · Epochs: `200`
🚀 Getting Started
1. Install dependencies
```bash
pip install numpy pillow
```
2. Check your project structure
Ensure `main.py` and the `Fruits Dataset` folder are in the same directory.
3. Run the program
```bash
python main.py
```
📋 Understanding the Output
The program reports:
Image processing: folders scanned and features extracted.
Model setup: network layers and configuration.
Training progress: loss printed every 20 epochs.
Evaluation: test loss and accuracy.
Transfer-function comparison: Hard Limit, Sigmoid, and Tanh outputs.
Sample predictions: outputs and true labels for the first 10 test samples.
📦 Dependencies
Package	Purpose
`numpy`	Matrix operations, neural-network calculations, and training
`Pillow` (`PIL`)	Loading and resizing images
---
<div align="center">
Built for learning the fundamentals of neural networks and image-based classification.
</div>
