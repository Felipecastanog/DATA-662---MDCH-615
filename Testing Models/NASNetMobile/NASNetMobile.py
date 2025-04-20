
import os
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.nasnet import preprocess_input

#Set seed
SEED = 3
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#File paths
train_dir = './train'
test_dir = './test'
metadata_path = './Chest_xray_Corona_Metadata.csv'

#Load metadata
metadata = pd.read_csv(metadata_path)

#Remove 'Stress-Smoking'
metadata = metadata[~(metadata['Label_1_Virus_category'] == 'Stress-Smoking')]

#Map labels
def map_labels(row):
    if row['Label'] == 'Normal':
        return 0
    elif row['Label'] == 'Pnemonia':
        if str(row['Label_1_Virus_category']).lower() == 'virus':
            return 1
        elif str(row['Label_1_Virus_category']).lower() == 'bacteria':
            return 2
    return np.nan

metadata['Label'] = metadata.apply(map_labels, axis=1)
metadata = metadata.dropna(subset=['Label'])

#read images
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype('float32')

#Load and preprocess image data
def load_images(meta_subset, img_dir):
    images, labels = [], []
    for _, row in meta_subset.iterrows():
        path = os.path.join(img_dir, row['X_ray_image_name'])
        img = preprocess_image(path)
        images.append(img)
        labels.append(row['Label'])
    return np.array(images), to_categorical(np.array(labels), num_classes=3)

train_meta = metadata[metadata['Dataset_type'] == 'TRAIN']
test_meta = metadata[metadata['Dataset_type'] == 'TEST']

train_images, train_labels = load_images(train_meta, train_dir)
test_images, test_labels = load_images(test_meta, test_dir)

#Split for validation set
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for train_idx, val_idx in sss.split(train_images, np.argmax(train_labels, axis=1)):
    X_train, X_val = train_images[train_idx], train_images[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]

#Data augmentation and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = datagen.flow(X_train, y_train, batch_size=32, seed=SEED)
val_generator = datagen.flow(X_val, y_val, batch_size=32, seed=SEED)

#Model building
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as keras

base_model = NASNetMobile(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)

base_model.trainable = False

#Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

#Compile and train
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weight_dict)

#Test preprocessing
test_images = preprocess_input(test_images)

#Initial evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Initial Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

#Fine-tuning
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weight_dict)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

#Test evaluation metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

y_pred_proba = model.predict(test_images)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(test_labels, axis=1)

final_accuracy = accuracy_score(y_true, y_pred)
micro_auc = roc_auc_score(y_true, y_pred_proba, average='micro', multi_class='ovr')
micro_precision = precision_score(y_true, y_pred, average='micro')
micro_recall = recall_score(y_true, y_pred, average='micro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

print(f"Final Evaluation Metrics:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Micro AUC: {micro_auc:.4f}")
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Micro F1 Score: {micro_f1:.4f}")


#Save the model
model.save("NASNetMobile_model.h5")