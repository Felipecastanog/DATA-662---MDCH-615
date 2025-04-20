import os
SEED = 11
os.environ['PYTHONHASHSEED'] = str(SEED)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import random
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.mixed_precision import set_global_policy
import shap 
import matplotlib.cm as cm
import json
from shap.maskers import Image as ShapImageMasker
from tensorflow.keras.utils import img_to_array, load_img

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, 
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(f"[CLUSTER ERROR] GPU configuration failed: {e}")

set_global_policy('mixed_float16')

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load and prepare data
df = pd.read_csv('Chest_xray_Corona_Metadata.csv')
df = df[~df['Label_1_Virus_category'].isin(['Stress-Smoking'])]
df['Final_Label'] = df.apply(
    lambda row: 'Normal' if row['Label'] == 'Normal' 
    else ('Pneumonia-Virus' if row['Label_1_Virus_category'] == 'Virus' 
          else 'Pneumonia-Bacteria'), axis=1)
test = df[df['Dataset_type'] == 'TEST']

# Load trained model
best_model = load_model('5Dense.keras')

# Prepare test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory='./test',
    x_col='X_ray_image_name',
    y_col='Final_Label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16,
    color_mode='rgb',
    shuffle=False,
    seed=SEED,
    classes=['Normal', 'Pneumonia-Bacteria','Pneumonia-Virus'],
    validate_filenames=True,
)

# Model evaluation
y_pred = best_model.predict(test_generator)
y_true_labels = test_generator.classes
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate metrics
micro_auc = roc_auc_score(test_generator.labels, y_pred, average='micro', multi_class='ovr')
micro_precision = precision_score(y_true_labels, y_pred_labels, average='micro')
micro_recall = recall_score(y_true_labels, y_pred_labels, average='micro')
micro_f1 = f1_score(y_true_labels, y_pred_labels, average='micro')

test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1 = best_model.evaluate(test_generator, verbose=0)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Micro AUC: {micro_auc:.4f}")
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

# Test Accuracy: 0.8654
# Test Loss: 0.5543
# Micro AUC: 0.9400
# Micro Precision: 0.8654
# Micro Recall: 0.8654
# Micro F1: 0.8654

