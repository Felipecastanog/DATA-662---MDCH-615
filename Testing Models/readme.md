# 📁 Testing Models

This folder contains all the experiments conducted to evaluate the performance of various pre-trained convolutional neural network (CNN) architectures using transfer learning for multi-class classification of chest X-ray images.

## 🧪 Purpose

Each subfolder corresponds to a specific CNN architecture used in testing. The goal was to compare their performance in classifying chest X-ray images into one of three classes: **Normal**, **Viral Pneumonia**, and **Bacterial Pneumonia**, using the CoronaHack dataset.

## 📂 Subdirectories

Each subdirectory includes the corresponding training and evaluation scripts, model checkpoints, and test results for the respective architecture:

- `Dense121`
- `Densenet201`
- `EfficientNetB0`
- `EfficientNet_V2M`
- `InceptionV3`
- `NASNetMobile`
- `ResNet101`
- `ResNet152`
- `VGG19`
- `Xception`

Each model was fine-tuned using a similar pipeline to ensure fair comparison, with variations in hyperparameters or architectural modifications where necessary.

## 📈 Evaluation

All models were evaluated using the same metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Micro AUC**

The results of these experiments are summarized in the final report and were used to select the best-performing model (EfficientNetV2-S), which was further refined and documented under the `Final Code` directory.

---

For reproducibility, the code in each subfolder can be executed independently. Ensure you have the dataset downloaded and placed in the correct directory as described in the [`Dataset`](../Dataset) folder.
