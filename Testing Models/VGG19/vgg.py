import os
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ======================== CONFIG ========================
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 10
PROJECT_PATH = os.getcwd()
CSV_TRAIN = os.path.join(PROJECT_PATH, "train_labels.csv")
CSV_TEST = os.path.join(PROJECT_PATH, "test_labels.csv")

# ===================== DATASET =====================
def create_generator(df, augment=False):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10 if augment else 0,
        zoom_range=0.1 if augment else 0,
        width_shift_range=0.1 if augment else 0,
        height_shift_range=0.1 if augment else 0,
        horizontal_flip=augment
    )
    df["full_path"] = df["image_path"].apply(lambda x: os.path.join(PROJECT_PATH, x))
    return datagen.flow_from_dataframe(
        df, x_col="full_path", y_col="label", target_size=(224, 224),
        class_mode="sparse", batch_size=BATCH_SIZE, shuffle=augment
    )

# ===================== MODEL LOADER =====================
def build_model(dropout=0.3, hidden_size=512):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden_size, activation="relu")(x)
    x = Dropout(dropout)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)

# ===================== TRAIN =====================
def train_model(model, train_gen, val_gen, epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)

# ===================== METRIC EVALUATION =====================
def evaluate_model(model, generator):
    y_true = generator.labels
    y_probs = model.predict(generator)
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="micro")
    prec = precision_score(y_true, y_pred, average="micro")
    rec = recall_score(y_true, y_pred, average="micro")
    try:
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_true), y_probs, multi_class="ovr", average="micro")
    except ValueError:
        auc = None
    print("\n📊 Evaluation Metrics:")
    print(f"Final Accuracy      : {acc:.4f}")
    print(f"Best Micro F1       : {f1:.4f}")
    print(f"Best Micro Precision: {prec:.4f}")
    print(f"Best Micro Recall   : {rec:.4f}")
    print(f"Best Micro AUC      : {auc:.4f}" if auc is not None else "Best Micro AUC      : Not computable")
    return acc

# ===================== OPTUNA =====================
def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.3, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    df = pd.read_csv(CSV_TRAIN)
    df["label"] = df["label"].astype(str)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    train_gen = create_generator(train_df, augment=True)
    val_gen = create_generator(val_df)

    model = build_model(dropout, hidden_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=NUM_EPOCHS,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)
    return evaluate_model(model, val_gen)

# ===================== MAIN =====================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\n Best Trial:")
    print(f"  Accuracy on validation set: {study.best_value:.4f}")
    best_params = study.best_trial.params
    for key, val in best_params.items():
        print(f"  {key}: {val}")

    print("\n Re-training best model on full training set and evaluating on test set...")
    train_df = pd.read_csv(CSV_TRAIN)
    test_df = pd.read_csv(CSV_TEST)
    train_df["label"] = train_df["label"].astype(str)
    test_df["label"] = test_df["label"].astype(str)

    train_gen = create_generator(train_df, augment=True)
    test_gen = create_generator(test_df)

    model = build_model(best_params["dropout"], best_params["hidden_size"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_model(model, train_gen, test_gen, NUM_EPOCHS)
    evaluate_model(model, test_gen)

    model.save(os.path.join(PROJECT_PATH, "best_model_vgg19.h5"))

