# Brain Tumour Detection using VGG16

# Overview

This project implements a Deep Learning–based brain tumour detection system using transfer learning with the VGG16 convolutional neural network.
The model classifies brain MRI images into multiple categories (Glioma, Meningioma, Pituitary Tumour, No Tumour) to assist in early diagnosis.

The system is built with TensorFlow/Keras, trained and evaluated on a labelled MRI dataset. GPU acceleration (Google Colab) is used to reduce training time and improve efficiency.

# Features

Transfer learning with VGG16 pretrained on ImageNet
Custom classification head with Dense + Dropout layers
Regularization to reduce overfitting
Runs efficiently on Google Colab GPU
Easy-to-use and extendable codebase

# Dataset

The dataset used for this project is publicly available on Kaggle:
 Brain Tumor MRI Dataset

It contains brain MRI images categorized into four classes:

Glioma
Meningioma
Pituitary Tumour
No Tumour

# Dataset Preparation

Images resized to 128×128 pixels
Normalized to [0,1] range
Shuffled to ensure unbiased training

# Methodology

Data Preprocessing
Resize and normalize MRI images
Shuffle and split into train/test sets
Optional augmentation for improved generalization
Model Architecture
Base Model: VGG16 (include_top=False, pretrained on ImageNet)
Custom Layers:Flatten → Dropout (0.3) → Dense (128, ReLU) → Dropout (0.2) → Dense (Softmax)

# Training

Optimizer: Adam (lr=1e-4)
Loss: Categorical Crossentropy
Metric: Accuracy
Hardware: GPU (Google Colab)

# Evaluation

Evaluated on test set using:
Accuracy Precision, Recall, F1-score

Overall Accuracy: 90.91%

Overall Precision: 91.44%

Overall Recall: 90.91%

Overall F1 Score: 91.08%
