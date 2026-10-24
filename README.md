# AdaFL: Adaptive Federated Learning for Brain Tumor Classification

## Overview
This project explores Adaptive Federated Learning (AdaFL) for brain tumor classification from MRI images using transfer learning with VGG16.

The approach trains a global model through multiple clients, combining locally trained model weights to support collaborative learning across distributed datasets.

## Objectives
- Apply transfer learning using a pretrained VGG16 model.
- Simulate multiple clients in a federated learning environment.
- Perform local model training and global model aggregation.
- Track classification performance across communication rounds.
- Evaluate predictions using a classification report and confusion matrix.

## Technologies Used
- Python
- TensorFlow / Keras
- VGG16
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Model Configuration
- Number of clients: 10
- Communication rounds: 3
- Client selection ratio: 0.5
- Learning rate: 0.0001
- Input image size: 128 × 128
- Model: VGG16 with a custom classification head

## Dataset
Brain MRI images are organized into class-specific folders for training and testing. Configure the dataset paths according to your environment before running the notebook.

## Repository Contents
- `ada_fl.ipynb` — notebook containing preprocessing, model construction, federated training, and evaluation code.

## How to Run
1. Open the notebook in Google Colab or Kaggle.
2. Download or attach the required brain MRI dataset.
3. Update the training and testing directory paths.
4. Install any missing dependencies.
5. Run the notebook cells in order.

## Evaluation
The notebook includes classification-report and confusion-matrix code for evaluating the model. Run the complete experiment to obtain the corresponding results.

## Disclaimer
This is an academic machine-learning project for experimentation and research. It is not intended for clinical diagnosis.
