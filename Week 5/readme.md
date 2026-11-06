# Digit Classification using LSTM

## Overview & Aim

The goal of this project is to build and train a sequence model—specifically a Long Short-Term Memory (LSTM) network—to classify grayscale images of handwritten digits (from 0 to 9) using TensorFlow and Keras.

While image classification is typically handled by Convolutional Neural Networks (CNNs), this project demonstrates an alternative educational approach: treating an image as a time-series sequence. By reading the image one row of pixels at a time, the LSTM learns the spatial dependencies between consecutive rows to predict the correct digit.

## Prerequisites & Setup Guide

To run this project, you will need a Python environment (Python 3.7 or higher is recommended) and a few standard data science libraries.

**Dependencies:**

* **TensorFlow (2.x):** The core deep learning framework used to build, train, and evaluate the model.

* **Matplotlib:** Used for visualizing the training and validation accuracy over time.

**Installation:**
You can install the necessary packages using `pip` in your terminal or command prompt:

```
pip install tensorflow matplotlib

```

**Running the Script:**

1. Save the provided Python code into a file named `lstm_mnist.py`.

2. Open your terminal and navigate to the directory where the file is saved.

3. Execute the script by running:

```
python lstm_mnist.py

```

## Dataset Used

This program uses the classic **MNIST** dataset loaded directly via `tensorflow.keras.datasets`.

* **Details:** The dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is 28x28 pixels.

* **Preprocessing:**

  * **Normalisation:** Pixel values (0-255) are divided by 255.0 to scale them to a `[0, 1]` range, ensuring stable and faster training.

  * **Sequence Reshaping:** Images are reshaped into `(28, 28)`. The LSTM interprets this as 28 **time steps** (the rows), where each step contains 28 **features** (the pixels in that row).

  * **One-Hot Encoding:** The integer labels (0-9) are converted into one-hot encoded vectors of length 10 using `to_categorical`.

## Network Architecture

The model uses a Keras `Sequential` architecture designed for sequence processing:

* **LSTM Layer:** A Recurrent Neural Network layer containing 128 memory units. It uses the `relu` activation function and takes an `input_shape` of `(28, 28)`. This layer processes the image row-by-row, maintaining an internal state of what it has seen so far.

* **Dense (Output) Layer:** A fully connected layer with 10 neurons (one for each possible digit). It uses the **Softmax** activation function to output a probability distribution across the 10 classes.

## Training & Expected Outcome

The model is compiled using the **Adam** optimizer and the **Categorical Crossentropy** loss function (ideal for one-hot encoded multi-class classification). It is trained for 10 epochs with a batch size of 64.

Upon running the script, you can expect:

1. **Console Outputs:** Verification of the reshaped data sizes and assertion checks confirming that the normalisation was successful.

2. **Model Summary:** A printed table detailing the parameters of the LSTM and Dense layers.

3. **Training Progress:** A progress bar for the 10 training epochs, followed by the final evaluation on the test set (typically achieving \~97-98% accuracy).

4. **Visualization:** A **Matplotlib chart** displaying the training and validation accuracy over time, helping to visualize the model's learning progress and check for overfitting.