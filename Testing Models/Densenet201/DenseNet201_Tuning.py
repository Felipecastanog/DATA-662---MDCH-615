import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import get_model, get_model_weights
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import optuna

# ======================== CONFIG ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 30  # Increased for deeper training
PROJECT_PATH = os.path.join(os.getcwd(), "Project")
MODEL_NAME = "Densenet201"  # Tuning this model

# ===================== DATASET =====================
class SimpleImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(PROJECT_PATH, self.data.loc[idx, 'image_path'])
        label = self.data.loc[idx, 'label']
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ===================== TRANSFORMS =====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2)], p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===================== MODEL LOADER =====================
def load_model(model_name, num_classes, dropout=0.3, hidden_size=512):
    weights_enum = get_model_weights(model_name)
    model = get_model(model_name, weights=weights_enum.DEFAULT)

    if hasattr(model, "classifier"):
        classifier = model.classifier
        in_features = classifier.in_features if isinstance(classifier, nn.Linear) else classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    else:
        raise ValueError("Model has no known classifier attribute (fc/classifier)")

    return model, weights_enum.DEFAULT

# ===================== TRAIN =====================
def train_model(model, optimizer, criterion, train_loader, scheduler=None):
    model.to(DEVICE)
    model.train()
    for epoch in range(NUM_EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()

# ===================== METRIC EVALUATION =====================
def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs_tensor = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs_tensor, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs_tensor.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="micro")
    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="micro")
    except ValueError:
        auc = None

    print("\n📊 Evaluation Metrics:")
    print(f"Final Accuracy      : {accuracy:.4f}")
    print(f"Best Micro F1       : {f1:.4f}")
    print(f"Best Micro Precision: {precision:.4f}")
    print(f"Best Micro Recall   : {recall:.4f}")
    print(f"Best Micro AUC      : {auc:.4f}" if auc is not None else "Best Micro AUC      : Not computable")
    print(f"Final Average Loss  : {avg_loss:.4f}")

    return accuracy

# ===================== OPTUNA =====================
def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.3, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    full_df = pd.read_csv(os.path.join(PROJECT_PATH, "train_labels.csv"))
    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
    train_loader = DataLoader(SimpleImageDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SimpleImageDataset(val_df, test_transform), batch_size=BATCH_SIZE, shuffle=False)

    model, _ = load_model(MODEL_NAME, NUM_CLASSES, dropout, hidden_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_model(model, optimizer, criterion, train_loader, scheduler)
    val_acc = evaluate_model(model, val_loader, criterion)
    return val_acc

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

    full_df = pd.read_csv(os.path.join(PROJECT_PATH, "train_labels.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_PATH, "test_labels.csv"))
    train_loader = DataLoader(SimpleImageDataset(full_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SimpleImageDataset(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False)

    model, _ = load_model(MODEL_NAME, NUM_CLASSES, best_params["dropout"], best_params["hidden_size"])
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_model(model, optimizer, criterion, train_loader, scheduler)
    test_acc = evaluate_model(model, test_loader, criterion)

    torch.save(model.state_dict(), os.path.join(PROJECT_PATH, "best_model_weights.pth"))
