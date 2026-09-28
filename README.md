# ViT100-Bayes-Method
# Multi-GPU Parallel Kernel Naive Bayes

GPU-accelerated Kernel Naive Bayes classifier using PyTorch, 1D Gaussian Kernel Density Estimation (KDE), and Multi-GPU parallel computation for lung X-ray classification.

## Pipeline

```text
Chest X-ray Images
        │
        ▼
Vision Transformer (ViT-Base/16)
        │
     768-D
        │
        ▼
Dense Feature Reducer (DFR)
        │
     100-D
        │
        ▼
Multi-GPU Parallel Kernel Naive Bayes
        │
        ▼
Predicted Class
```

## Requirements
```python
pip install numpy scipy scikit-learn torch torchvision timm
```
## Features
```text
PyTorch-based Kernel Naive Bayes
1D Gaussian Kernel Density Estimation
GPU acceleration
Multi-GPU support using torch.nn.DataParallel
Scott's and Silverman's bandwidth estimation
Log-Sum-Exp for numerical stability
Class prior probability estimation
Probability prediction
Classification report
Training and inference time measurement
```

## Input Features
The classifier receives the 100-dimensional features extracted from the ViT-DFR model.
```text

ViT-Base/16
     │
     ▼
768-dimensional features
     │
     ▼
DFR Block
     │
     ├── 768 → 512
     ├── 512 → 256
     └── 256 → 100
     │
     ▼
100-dimensional features

```
The DFR block uses:

Linear layers
Layer Normalization
GELU activation
Dropout


# ViT100 Feature Extraction

```python
import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import time
# ==========================================================
# 1. PATH AND PARAMETERS
# ==========================================================
TRAIN_DIR = "/kaggle/input/datasets/tonytoanpham/training/train"
VAL_DIR   = "/kaggle/input/datasets/tonytoanpham/testing/test"  # Dùng làm Validation set
CLASSES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis"]
IMG_SIZE = 224    
BATCH_SIZE = 32   
EPOCHS = 30        
NUM_CLASSES = len(CLASSES)
FEATURE_DIM = 100 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
print(f"Device: {device} | Number of GPUs available: {gpu_count}")
# ----------------------------------------------------------
# Data Augmentation & Normalization
# ----------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])
# ==========================================================
# 2. DATASET & DATALOADER
# ==========================================================
class LungXrayDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.samples = []
        self.classes = classes
        self.transform = transform
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Folder not found: {class_path}")
                continue
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, idx))
        print(f"Loaded {len(self.samples)} images from {root_dir}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
# Datasets
train_dataset = LungXrayDataset(TRAIN_DIR, CLASSES, train_transform)
val_dataset   = LungXrayDataset(VAL_DIR, CLASSES, val_transform)
# DataLoader phục vụ huấn luyện
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True, 
    drop_last=True,
    persistent_workers=True)
# DataLoader phục vụ Đánh giá Validation
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True)
# DataLoader phục vụ Trích xuất Feature Train (Không Augmentation)
eval_train_dataset = LungXrayDataset(TRAIN_DIR, CLASSES, val_transform)
eval_train_loader  = DataLoader(eval_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
# ==========================================================
# 3. DFR BLOCK & MODEL ARCHITECTURE
# ==========================================================
class DFRBlock(nn.Module):
    def __init__(self, in_features=768, target_dim=100, dropout_rate=0.3):
        super().__init__()
        self.dfr = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, target_dim),
            nn.LayerNorm(target_dim),
            nn.GELU())
    def forward(self, x):
        return self.dfr(x)
class ViT16_DFR_Model(nn.Module):
    def __init__(self, num_classes=4, target_feature_dim=100):
        super().__init__()
        MODEL_NAME = "vit_base_patch16_224"
        print(f"--> Building Model with Backbone: {MODEL_NAME}")
        self.vit = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0)
        self.in_features = self.vit.num_features 
        self.dfr_block = DFRBlock(in_features=self.in_features, target_dim=target_feature_dim)
        self.classifier = nn.Linear(target_feature_dim, num_classes)
    def forward(self, x):
        feat_768 = self.vit(x)
        feat_100 = self.dfr_block(feat_768)
        out = self.classifier(feat_100)
        return out
    def extract_features(self, x):
        feat_768 = self.vit(x)
        feat_100 = self.dfr_block(feat_768)
        return feat_100
model = ViT16_DFR_Model(num_classes=NUM_CLASSES, target_feature_dim=FEATURE_DIM)
# Freeze backbone ngoại trừ 4 blocks cuối
for param in model.vit.parameters():
    param.requires_grad = False
if hasattr(model.vit, 'blocks'):
    for block in model.vit.blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True
model = model.to(device)
if gpu_count > 1:
    model = nn.DataParallel(model)
# Tách biệt Learning Rate
backbone_params = []
head_params = []
base_model = model.module if isinstance(model, nn.DataParallel) else model
for name, param in base_model.named_parameters():
    if param.requires_grad:
        if "vit" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-4}], weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
# ==========================================================
# 4. TRAINING & VALIDATION LOOP
# ==========================================================
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}
print("\n--- Starting Model Training ---")
for epoch in range(EPOCHS):
    # --- TRAIN PHASE ---
    model.train()
    train_correct, train_total = 0, 0
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
    scheduler.step()
    epoch_train_loss = running_train_loss / train_total
    epoch_train_acc = 100.0 * train_correct / train_total
    # --- VALIDATION PHASE ---
    model.eval()
    val_correct, val_total = 0, 0
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    epoch_val_loss = running_val_loss / val_total
    epoch_val_acc = 100.0 * val_correct / val_total
    # Lưu kết quả theo từng epoch
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
          f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

print("\nTraining finished!")
# ==========================================================
# 5. PLOT TRAINING & VALIDATION METRICS
# ==========================================================
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(14, 5))
# Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='red', linestyle='--', linewidth=2)
plt.title('Training and Validation Loss', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
plt.title('Training and Validation Accuracy', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
# ==========================================================
# 6. FEATURE EXTRACTION
# ==========================================================
def extract_dense_features(model, loader):
    model.eval()
    features, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with autocast():
                if isinstance(model, nn.DataParallel):
                    z = model.module.extract_features(images)
                else:
                    z = model.extract_features(images)

            features.append(z.cpu().numpy())
            labels_all.append(labels.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels_all, axis=0)

print("\nExtracting 100-D features...")
X_train_100, y_train = extract_dense_features(model, eval_train_loader)
X_val_100, y_val     = extract_dense_features(model, val_loader)
print("X_train_100 shape:", X_train_100.shape)
print("X_val_100 shape  :", X_val_100.shape)
```
# Output
```text
<p align="center">
  <img 
    src="figures/framework.png" 
    alt="ViT-DFR and Kernel-based Naive Bayes Framework"
    width="900"
  >
</p>
```
```text
X_train_100 shape: (6465, 100)
X_val_100 shape  : (1632, 100)
```

## Bandwidth Estimation

The implementation supports Scott's rule:
```python
bw = (n ** (-1/5)) * std
bw = ((4 / 3 * n) ** (-1/5)) * std
knb = FastParallelKernelNaiveBayes(
    bandwidth_method='silverman')
```

## Multi-GPU Computation

The available GPUs are automatically detected:

```python

```









