# 📁 Dataset

This folder contains metadata and configuration files used to train, validate, and test the models developed in the project:

**Machine Learning Classification Using Transfer Learning and X-Ray Images**

## 📄 Contents

- `Chest_xray_Corona_Metadata.csv`: Pre-processed metadata file used for filtering, labeling, and dataset splitting.
- `label_mappings.json`: Maps original labels to the final three-class setup (`Normal`, `Viral Pneumonia`, `Bacterial Pneumonia`).
- `train_labels.csv`: Image filenames and class labels used for training the models.
- `test_labels.csv`: Image filenames and class labels used for model evaluation.

## 🖼️ Chest X-Ray Images

Due to file size limitations, the full set of chest X-ray image files could not be uploaded to this GitHub repository.

However, all images used for training, validation, and testing can be accessed and downloaded from Kaggle:

👉 [CoronaHack - Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset/data)

Once downloaded, please place the image files into the following directory relative to the project root:

