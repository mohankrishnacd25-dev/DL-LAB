# The XOR Problem: From a Single Neuron to a Multi-Layer Perceptron

## Overview & Aim
The goal of this program is to demonstrate how a Multi-Layer Perceptron (MLP) can successfully solve the XOR (exclusive OR) problem. 

Historically, the XOR problem was a significant hurdle in early artificial intelligence. A standard single-layer perceptron cannot solve XOR because the classes (0 and 1) are not linearly separable—you cannot draw a single straight line to separate the true outputs from the false outputs. By adding a hidden layer, we allow the neural network to map the inputs into a higher-dimensional space where they become linearly separable, allowing the model to learn this non-linear logic.

## Dataset Used
The dataset represents the standard XOR truth table, hardcoded using NumPy arrays. In an XOR gate, the output is true (1) only if the inputs are different.

* **Inputs (X):** 
  * `[0, 0]`
  * `[0, 1]`
  * `[1, 0]`
  * `[1, 1]`

* **Target Outputs (y):** 
  * `[0]`
  * `[1]`
  * `[1]`
  * `[0]`

## Model Architecture
The program uses `tensorflow.keras` to build a Sequential multi-layer neural network with the following architecture:
1. **Input Layer:** Accepts 2 features (the two binary inputs).
2. **Hidden Layer:** A Dense layer with 8 neurons utilizing the `ReLU` (Rectified Linear Unit) activation function to introduce non-linearity.
3. **Output Layer:** A single neuron Dense layer using the `sigmoid` activation function to output a probability between 0 and 1.

## Prerequisites
To run this script, you will need Python installed along with the following libraries:
* `numpy`
* `tensorflow`

You can install them via pip:
```bash
pip install numpy tensorflow
```

## Results & Output
The model is compiled using `binary_crossentropy` as the loss function (ideal for binary classification) and Stochastic Gradient Descent (`SGD`) with a learning rate of 0.1. 

After training for 1000 epochs, the network reliably converges. 
* **Accuracy:** The model typically achieves **100% accuracy** during the evaluation phase.
* **Predictions:** When prompted to predict the outputs for the original inputs, the network outputs probabilities that, when rounded, perfectly match the target labels `[0, 1, 1, 0]`.