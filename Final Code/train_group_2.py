# ========== IMPORTS ==========
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, get_model

# ========== CONFIGURATION ==========
# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Setup computation device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 6.240039102200668e-05
HIDDEN_SIZE = 256
DROPOUT = 0.4111310329735102
MODEL_NAME = "efficientnet_v2_s"

# ========== DATASET DEFINITION ==========
class SimpleImageDataset(Dataset):
    """Custom dataset class for chest X-ray images from metadata CSV."""
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.loc[idx, 'X_ray_image_name'])
        image = Image.open(img_name).convert("RGB")
        label = int(self.df.loc[idx, 'label'])
        if self.transform:
            image = self.transform(image)
        return image, label

# ========== LABEL MAPPING FUNCTION ==========
def map_label(row):
    """Maps string labels from metadata to numeric codes."""
    if row["Label"] == "Normal":
        return 0
    elif row["Label_1_Virus_category"].lower() == "virus":
        return 1
    elif row["Label_1_Virus_category"].lower() == "bacteria":
        return 2
    else:
        return None

# ========== TRANSFORMS ==========
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== MODEL LOADER ==========
def load_model():
    """Builds model using architecture and hyperparameters from config."""
    model = get_model(MODEL_NAME, weights=EfficientNet_V2_S_Weights.DEFAULT)

    # Replace classifier or fc layers depending on model type
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        )
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        )
    else:
        raise ValueError("Unsupported model architecture")

    return model

# ========== TRAIN FUNCTION ==========
def train_model(model, optimizer, criterion, train_loader):
    """Standard training loop over the dataset."""
    model.to(DEVICE)
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

# ========== MAIN ==========
if __name__ == "__main__":
    # Read and filter metadata
    metadata = pd.read_csv("Chest_xray_Corona_Metadata.csv")
    metadata = metadata[~metadata["Label_1_Virus_category"].isin(["Stress-Smoking"])]
    metadata = metadata[metadata["Label"].isin(["Normal", "Pnemonia"])]

    # Map labels to integer codes
    metadata["label"] = metadata.apply(map_label, axis=1)

    # Filter train set and add image paths
    metadata = metadata[metadata["label"].notnull()]
    train_df = metadata[metadata["Dataset_type"] == "TRAIN"]

    # Load train dataset
    train_dataset = SimpleImageDataset(train_df, root_dir="./train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load model
    model = load_model()
    model.to(DEVICE)

    # Training on best hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training on GPU..." if DEVICE.type == 'cuda' else "Starting training on CPU...")
    train_model(model, optimizer, criterion, train_loader)

    # Save trained model
    torch.save(model.state_dict(), "best_model_group_2.pth")
    print("Model saved as best_model_group_2.pth")