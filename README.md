# Medical MNIST Classification with CNN
<img width="534" height="176" alt="image" src="https://github.com/user-attachments/assets/b52b8ce8-88b2-40d0-a0be-c7c43d7ffa0e" />

This repository contains a complete pipeline for classifying medical images using a Convolutional Neural Network (CNN). The project demonstrates the transition from a raw data to a high-performance TensorFlow data pipeline, concluding with a trained model capable of multi-class medical image categorization. The model achieves high accuracy on the test set (99.92%).

## Dataset Source

The data used in this project is the **Medical MNIST** dataset, which can be found on Kaggle:
[https://www.kaggle.com/datasets/andrewmvd/medical-mnist/data](https://www.kaggle.com/datasets/andrewmvd/medical-mnist/data)

This dataset consists of 58,954 images across 6 classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, and HeadCT.
## Project Overview

The goal of this project is to automate the classification of various medical imaging modalities (such as X-rays or MRIs) by leveraging deep learning. The system uses a CSV file as a centralized map to locate images on disk, processes them into a standardized format, and trains a sequential CNN to achieve high classification accuracy.

## Technical Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing & Splitting:** Scikit-Learn
* **Visualization:** Matplotlib


## Data Pipeline

The project implements a "Lazy-Loading" architecture to ensure memory efficiency, allowing the model to train on large datasets without loading all images into RAM simultaneously.

1.  **Label Encoding:** Categorical string labels from the CSV are converted into integer codes for neural network compatibility.
2.  **Dataset Splitting:** The initial dataframe is split into training (80%) and testing (20%) sets using `train_test_split`.
3.  **Image Preprocessing:** A custom mapping function performs the following operations:
    * Reads raw files from disk.
    * Decodes images into grayscale tensors.
    * Resizes images to a uniform 64x64 resolution.
    * Normalizes pixel values to a [0, 1] range.
4.  **Performance Optimization:** The pipeline utilizes `tf.data.AUTOTUNE` for prefetching and batching to minimize CPU-to-GPU bottlenecks.



## Model Architecture

The model is a Sequential CNN designed to extract spatial features through increasing levels of abstraction.

* **Convolutional Layers:** Three layers using 3x3 kernels and ReLU activation to identify patterns ranging from simple edges to complex organic textures.
* **Pooling Layers:** Max-pooling layers (2x2) to reduce spatial dimensions and increase translation invariance.
* **Fully Connected Layers:** A dense layer with 64 units followed by an output layer using Softmax activation.
* **Loss Function:** Sparse Categorical Crossentropy, ideal for integer-labeled multi-class tasks.


## How to Use

### Prerequisites
Install the required dependencies:
```bash
pip install tensorflow pandas scikit-learn matplotlib
