# MLP from Scratch: Batch Gradient Descent vs. Stochastic Gradient Descent

## Overview & Aim

The goal of this programme is to build a Multi-Layer Perceptron (MLP) neural network entirely from scratch using only NumPy. By avoiding high-level frameworks like TensorFlow or PyTorch, this project demonstrates the underlying mathematics of forward propagation and backpropagation.

Additionally, the programme aims to classify non-linear data while comparing the performance and convergence behaviours of two fundamental optimisation strategies:

1. **Batch Gradient Descent:** Calculating the gradient over the entire dataset before making a single weight update per epoch.

2. **Stochastic Gradient Descent (SGD):** Shuffling the data and updating the weights multiple times per epoch using smaller mini-batches.

## Dataset Used

This program uses the `make_moons` synthetic dataset from the `scikit-learn` library, which is ideal for testing non-linear classifiers.

* **Details:** It generates 400 samples of 2D data points that form two interleaving half-circles (moons).

* **Preprocessing:** A noise level of 20% is added to scatter the points slightly, making the classification task more challenging.

* **Standardisation:** The input features are normalised by subtracting the mean and dividing by the standard deviation. This ensures zero mean and unit variance, which helps the network converge faster and prevents vanishing/exploding gradients.

## Network Architecture

The neural network is dynamically built based on the `sizes` array.

* **Input Layer:** 2 neurones (representing the 2D coordinates of the dataset).

* **Hidden Layers:** Two hidden layers, each with 16 neurones. These use the **ReLU** (Rectified Linear Unit) activation function to introduce non-linearity.

* **Output Layer:** 1 neurone using the **Sigmoid** activation function ($\sigma(z) = \frac{1}{1 + e^{-z}}$) to output a probability between 0 and 1 for binary classification.

* **Initialisation:** The weights are initialised using **He Initialisation(),** which is mathematically optimised for networks using ReLU activations.

## Results

After training the neural network for 200 epochs using a learning rate of 0.5, the network successfully learns the complex, non-linear boundaries of the moon dataset.

* **Batch Gradient Descent** achieves a high accuracy, typically around **96.50%**.

* **Stochastic Gradient Descent (SGD)** with mini-batches slightly outperforms it, reaching around **97.75%**.

**Conclusion:** The results demonstrate that Mini-Batch SGD generally converges faster and yields better generalisation in fewer epochs. Because it updates the weights multiple times per epoch and introduces slight noise into the gradient calculations, it is better equipped to escape local minima in the loss landscape compared to standard Batch GD.