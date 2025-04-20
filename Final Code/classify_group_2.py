# ========== IMPORTS ==========
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import get_model, get_model_weights
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ========== CONFIG ==========
# Setup computation device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if it is available

# Define constants
NUM_CLASSES = 3 # Normal, Pneumonia-Virus, Pneumonia-Bacteria
BATCH_SIZE = 32
MODEL_NAME = "efficientnet_v2_s"
HIDDEN_SIZE = 256
DROPOUT = 0.4111310329735102

# Define file paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_PATH, f"best_model_group_2.pth")
TEST_FOLDER = os.path.join(PROJECT_PATH, "test")

# Map numeric labels to class names
LABEL_MAP = {0: "Normal", 1: "Pneumonia-Virus", 2: "Pneumonia-Bacteria"}

# ========== TRANSFORMS ==========
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== DATASET ==========
class SimpleImageDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading test images and labels."""
    def __init__(self, df, transform=None):
        self.data = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.loc[idx, 'image_path']
        label = self.data.loc[idx, 'label']
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path
    
# ========== LABEL MAPPING FUNCTION ==========   
def map_label(row):
    """Maps text labels from metadata to numerical codes."""
    if row["Label"] == "Normal":
        return 0
    elif row["Label_1_Virus_category"] == "virus":
        return 1
    elif row["Label_1_Virus_category"] == "bacteria":
        return 2
    else:
        return None
    
# ========== MODEL LOADER ==========
def load_model():
    """Builds model using architecture and hyperparameters from config."""

    model = get_model(MODEL_NAME, weights=None)

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
# ========== EVALUATION ==========
def evaluate_model(model, data_loader, criterion):
    """Evaluates model on test set and prints metrics."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
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

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="micro")
    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="micro")
    except ValueError:
        auc = None

    # Display results
    print("\n📊 Evaluation Results:")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"AUC           : {auc:.4f}" if auc is not None else "AUC           : Not computable")

# ======== GRAD-CAM =========
def generate_gradcam(model, image_tensor, raw_image, class_idx, save_path):
    """Generates and saves Grad-CAM heatmap for a given image and class prediction."""
    target_layers = [model.features[-1]] if hasattr(model, "features") else [list(model.children())[-2]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)[0, :]

    rgb_image = raw_image.permute(1, 2, 0).cpu().numpy()
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    plt.imshow(cam_image)
    plt.axis('off')
    plt.title(f"Predicted: {LABEL_MAP[class_idx]}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM to {save_path}")

# ========== MAIN ==========
if __name__ == "__main__":

    # Read and filter metadata
    metadata = pd.read_csv(os.path.join(PROJECT_PATH, "Chest_xray_Corona_Metadata.csv"))
    metadata = metadata[~metadata["Label_1_Virus_category"].isin(["Stress-Smoking"])]
    metadata = metadata[metadata["Label"].isin(["Normal", "Pnemonia"])]
    metadata["Label_1_Virus_category"] = metadata["Label_1_Virus_category"].str.lower()

    # Map labels to integer codes
    metadata["label"] = metadata.apply(map_label, axis=1)

    # Filter test set and add image paths
    test_df = metadata[metadata["Dataset_type"] == "TEST"].copy()
    test_df["image_path"] = test_df["X_ray_image_name"].apply(lambda x: os.path.join(TEST_FOLDER, x))

    # Load test dataset
    test_dataset = SimpleImageDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load weights
    model = load_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # Evaluate on test data
    criterion = nn.CrossEntropyLoss()
    print("Evaluating on GPU..." if DEVICE.type == 'cuda' else "Evaluating on CPU...")
    evaluate_model(model, test_loader, criterion)

    # Grad-CAM visualization for 5 random test images
    raw_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    sampled_indices = random.sample(range(len(test_dataset)), 5)

    fig, axs = plt.subplots(5, 2, figsize=(8, 20))
    model.eval()

    for row, idx in enumerate(sampled_indices):
        image, true_label, path = test_dataset[idx]
        input_tensor = test_transform(Image.open(path).convert("RGB"))
        raw_tensor = raw_transform(Image.open(path).convert("RGB"))
        input_tensor = input_tensor.to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()

        target_layers = [model.features[-1]] if hasattr(model, "features") else [list(model.children())[-2]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        rgb_image = raw_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        axs[row][0].imshow(rgb_image)
        axs[row][0].set_title(f"Original: {LABEL_MAP[true_label]}")
        axs[row][0].axis('off')

        axs[row][1].imshow(cam_image)
        axs[row][1].set_title(f"Pred: {LABEL_MAP[pred_class]}")
        axs[row][1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_PATH, "gradcam_panel.png"))
    print("Saved combined Grad-CAM figure to gradcam_panel.png")
