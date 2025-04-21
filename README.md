# DATA-662 - MDCH-615

# 🧠 Final Project — Data 622

**Machine Learning Classification Using Transfer Learning and X-Ray Images**

This project explores the use of **transfer learning** for multi-class classification of chest X-ray images into three clinically meaningful categories: **Normal**, **Viral Pneumonia**, and **Bacterial Pneumonia**. Leveraging the publicly available **CoronaHack Chest X-Ray Dataset**, we benchmarked various state-of-the-art CNN architectures and fine-tuned them for medical image classification tasks.

> 📌 Best model: **EfficientNetV2-S**, achieving **92.79% test accuracy** and **0.9780 micro-AUC**.

---

## 📂 Repository Structure

```bash
Final_Project_Data622/
│
├── Dataset/               # Metadata and label files (images downloaded separately from Kaggle)
├── Final Code/            # Final version of training and classification scripts using EfficientNetV2-S
├── Testing Models/        # Experiments with multiple pre-trained CNN architectures
│   ├── ResNet101/
│   ├── Dense121/
│   ├── Xception/
│   └── ... (etc.)
│
├── README.md              # Main project README (this file)
