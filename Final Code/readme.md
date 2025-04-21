# 📁 Final Code

This folder contains the final Python scripts used to train and evaluate the best-performing deep learning model in our project:

**Machine Learning Classification Using Transfer Learning and X-Ray Images**

## 📄 Files

- `train_group_2.py`: Script used to train the model on the processed dataset. It includes data loading, augmentation, model setup (EfficientNetV2-S), fine-tuning, and logging of training metrics.
- `classify_group_2.py`: Script for evaluating the trained model on the test dataset. It generates predictions, computes performance metrics (accuracy, precision, recall, F1-score, AUC), and produces Grad-CAM visualizations to interpret model decisions.

## 🧠 Model Summary

These scripts implement the final version of our transfer learning pipeline using **EfficientNetV2-S**, selected based on its high performance in multi-class classification of chest X-rays. The pipeline includes:

- Transfer learning with fine-tuning
- Use of class weights for handling imbalance
- Image augmentation (cropping, flipping, rotation, color jitter)
- Grad-CAM for model explainability
- Evaluation with micro-averaged metrics

## 📝 Usage

Make sure to prepare your dataset according to the instructions in the [`Dataset`](../Dataset) folder and install the dependencies specified in the project’s main `requirements.txt`.

