"""
================================================================
 KAGGLE TRAINING NOTEBOOK — Emotion Prediction (FER2013)
 v2 — Architecture matches backend EmotionModel exactly
================================================================
Instructions:
  1. Create a new Kaggle Notebook from FER2013 dataset page
  2. Enable GPU: Settings → Accelerator → GPU T4 x2
  3. Paste this entire script into a code cell and Run All

Output: 'emotion_model.pth' in /kaggle/working/
Place in: PROJECT/data/weights/emotion_model.pth
================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 64
NUM_EPOCHS  = 30
LR          = 1e-4
PATIENCE    = 5
EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# ── Dataset ───────────────────────────────────────────────────
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        for label_idx, emotion in enumerate(EMOTIONS):
            folder = self.root / emotion
            if folder.exists():
                for img_path in folder.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((img_path, label_idx))
        print(f"[{split}] Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms ────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DATA_ROOT = "/kaggle/input/datasets/msambare/fer2013"

train_set    = FER2013Dataset(DATA_ROOT, "train", train_transform)
val_set      = FER2013Dataset(DATA_ROOT, "test",  val_transform)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model — EXACT match to backend EmotionNet ─────────────────
# Backend: backbone.fc = nn.Sequential(Dropout(0.4), Linear(512, 7))
class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

model     = EmotionNet(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler    = torch.amp.GradScaler('cuda')

# ── Training ──────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
OUTPUT_PATH      = "/kaggle/working/emotion_model.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            val_loss    += loss.item()
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc   = val_correct / len(val_set)
    print(f"Epoch {epoch:03d} | Train Loss: {running_loss/len(train_loader):.4f} "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # ✅ Save in backend-compatible checkpoint format
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":         val_loss,
        }, OUTPUT_PATH)
        print(f"  ✅ Saved best model → {OUTPUT_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

print(f"\n🎉 Training complete! Best Val Loss: {best_val_loss:.4f}")
print(f"   Download: {OUTPUT_PATH}")
print(f"   Place in: PROJECT/data/weights/emotion_model.pth")
