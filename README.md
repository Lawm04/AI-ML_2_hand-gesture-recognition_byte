# Hand Gesture Recognition

## Project Overview

This project uses a Convolutional Neural Network (CNN) to classify
hand gestures from infrared images in the LeapGestRecog dataset.

## Objectives

- Prepare and inspect an image dataset.
- Train a CNN to classify hand gestures.
- Evaluate the model on subjects not used for training.
- Generate predictions and visualize evaluation results.

## Dataset

Dataset: LeapGestRecog

Source: https://www.kaggle.com/datasets/gti-upm/leapgestrecog

Gesture labels are derived from the gesture-folder names.
Images are resized to 64 x 64 pixels and converted to grayscale.

## Technologies

- Python
- TensorFlow / Keras
- NumPy and Pandas
- Pillow
- Matplotlib
- scikit-learn

## Project Structure

- data/raw/: Original dataset images
- data/processed/: Dataset split manifest
- src/: Dataset preparation, training, prediction, and demo scripts
- models/: Saved model and class labels
- results/: Evaluation metrics and visualizations

## Setup

Create and activate a Python virtual environment, then install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Workflow

1. Prepare the dataset:

       python src/prepare_dataset.py

2. Train and evaluate the model:

       python src/train.py

3. Predict a gesture from an image:

       python src/predict.py "path/to/image.png"

4. Generate the prediction gallery:

       python src/create_demo.py

## Evaluation

See results/classification_report.txt for evaluation metrics
and results/confusion_matrix.png for the confusion matrix.

Record your actual test accuracy and observations here after training.

## Limitations

The model is trained on infrared images and may not perform well
on ordinary webcam images. Test performance can also vary across subjects.

## Author

Your Name
