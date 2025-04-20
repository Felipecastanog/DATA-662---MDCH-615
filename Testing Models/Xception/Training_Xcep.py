import os

SEED = 11 
os.environ['PYTHONHASHSEED'] = str(SEED)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.regularizers import l2
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import random

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Enable mixed precision
set_global_policy('mixed_float16')
print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy()}")

# GPU memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit= 6144)]
            )
            print('Ready')
    except RuntimeError as e:
        print(f"[CLUSTER ERROR] GPU configuration failed: {e}")

# Data Loading
df = pd.read_csv('/home/carlos.arteagadeanda/Chest_xray_Corona_Metadata.csv')
df = df[~df['Label_1_Virus_category'].isin(['Stress-Smoking'])]
df['Final_Label'] = df.apply(
    lambda row: 'Normal' if row['Label'] == 'Normal' 
    else ('Pneumonia-Virus' if row['Label_1_Virus_category'] == 'Virus' 
          else 'Pneumonia-Bacteria'), axis=1)

train = df[df['Dataset_type'] == 'TRAIN'] 

# Class Weights

unique_classes = np.unique(train['Final_Label'])  # Get sorted unique classes as numpy array
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train['Final_Label'])
class_weights_dict = dict(zip(unique_classes, class_weights))

# Data Generators
TARGET_SIZE = (299, 299)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.95, 1.05],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='./train',
    x_col='X_ray_image_name',
    y_col='Final_Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
    classes=unique_classes.tolist()
)

# Model Definition
base_model = keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3),
    pooling=None
)
base_model.trainable = False

inputs = Input(shape=(299, 299, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Increased dropout
x = Dense(512, kernel_regularizer=l2(1e-4))(x)  # Added L2 regularization
x = BatchNormalization()(x)
x = Activation(tf.nn.swish)(x)  # Swish often outperforms ReLU
x = Dropout(0.4)(x)
outputs = Dense(3, activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs)

# Initial Training
print("Starting initial training...")
initial_lr = 0.0001
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=len(train_generator) * 25
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall'),
        tf.keras.metrics.F1Score(name='f1', average='micro')
    ]
)

history = model.fit(
    train_generator,
    epochs=25,
    class_weight=class_weights_dict
)

# Fine-Tuning
print("Starting fine-tuning...")
for layer in base_model.layers[-50:]:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = l2(1e-4)
    layer.trainable = True

finetune_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.00001,
    decay_steps=len(train_generator) * 10
)

model.compile(
    optimizer=Adam(learning_rate=finetune_lr),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        'accuracy',
        AUC(name='auc', curve='ROC'),
        Precision(name='precision'),
        Recall(name='recall'),
        tf.keras.metrics.F1Score(name='micro_f1', average='micro')
    ]
)

history_finetune = model.fit(
    train_generator,
    initial_epoch=history.epoch[-1] + 1,
    epochs=35,
    class_weight=class_weights_dict
)

model.save('2Xcep.keras')
