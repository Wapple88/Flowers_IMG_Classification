# Flower Classification with CNN using TensorFlow

## Overview
This Jupyter Notebook demonstrates a deep learning pipeline for image classification using TensorFlow and TensorFlow Datasets (TFDS). Specifically, it uses a Convolutional Neural Network (CNN) to classify images from the Oxford Flowers 102 dataset, which contains 102 categories of flowers. The notebook covers data preprocessing, model definition, training with data augmentation, and model evaluation.

## Requirements
- Python 3.x
- TensorFlow 2.x
- TensorFlow Datasets
- Keras (part of TensorFlow 2.x)
- NumPy
- Matplotlib
- Google Colab (optional, but configured for use with Google Drive)

Code Structure
1. Data Loading and Preprocessing
Loads the Oxford Flowers 102 dataset from TFDS.
Splits dataset into training, validation, and test sets, with details about classes and image shapes printed.
Preprocessing functions:
crop_and_resize_test: Center-crops images to a square and resizes to 224x224 pixels.
crop_and_resize_train: Randomly crops and applies augmentations to training data.
Batches data for efficient processing with caching and prefetching.
2. Data Augmentation
Adds augmentation layers for the training data:
Random contrast and brightness adjustments
Random horizontal and vertical flips
Random rotation and zoom
3. Model Architecture
Defines a CNN model with:
Conv2D and MaxPooling2D layers for feature extraction.
Dense layers with dropout for regularization.
Softmax output layer for multiclass classification.
Model summary and architecture diagram are generated.
4. Model Compilation and Training
Compiles model with:
Adam optimizer (learning rate 0.0001)
Sparse categorical cross-entropy loss
Accuracy metric
Uses callbacks to:
Save the best model based on validation accuracy
Apply early stopping
Reduce learning rate on validation accuracy plateau
5. Evaluation and Visualization
Evaluates the model on the test set.
Plots training and validation accuracy/loss over epochs to monitor performance.
Saves the model summary and architecture diagram.
6. Saving the Model
Saves the trained model to Google Drive for future use.
Usage
Run the cells sequentially in a Jupyter environment (e.g., Google Colab). Once training completes, visualize results through plots and save the final model to Google Drive.

Results
The final model's accuracy on the test set can be evaluated and visualized through the generated accuracy and loss plots.

