"""
================================================================
 KAGGLE TRAINING NOTEBOOK — Emotion Prediction (FER2013)
 v3 — Class-weighted loss, LR scheduler, strong augmentation
================================================================
Instructions:
  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
  2. Click "New Notebook" on the dataset page
  3. Enable GPU: Settings → Accelerator → GPU T4 x2
  4. Paste this entire script into a code cell and Run All

Output: 'emotion_model.pth' in /kaggle/working/
Place in: PROJECT/data/weights/emotion_model.pth
================================================================
"""

import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 64
NUM_EPOCHS  = 30
LR          = 1e-4
PATIENCE    = 7

# ⚠️ IMPORTANT: This order MUST match the backend (app/core/config.py)
EMOTIONS    = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)

# ── Dataset ───────────────────────────────────────────────────
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.labels = []
        for label_idx, emotion in enumerate(EMOTIONS):
            folder = self.root / emotion
            if folder.exists():
                for img_path in folder.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((img_path, label_idx))
                        self.labels.append(label_idx)
        print(f"  [{split}] Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms (STRONGER augmentation for better generalization) ─
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Auto-detect Kaggle dataset path ──────────────────────────
POSSIBLE_PATHS = [
    "/kaggle/input/fer2013",
    "/kaggle/input/fer2013/fer2013",
    "/kaggle/input/datasets/msambare/fer2013",
]
DATA_ROOT = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and (os.path.exists(os.path.join(p, "train")) or os.path.exists(os.path.join(p, "test"))):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    # Try to find it
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            DATA_ROOT = p
            break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/input/fer2013"
print(f"📂 Data root: {DATA_ROOT}")

train_set    = FER2013Dataset(DATA_ROOT, "train", train_transform)
val_set      = FER2013Dataset(DATA_ROOT, "test",  val_transform)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Log class distribution ───────────────────────────────────
print("\n📊 Class distribution (train):")
counts = Counter(train_set.labels)
for i, name in enumerate(EMOTIONS):
    c = counts.get(i, 0)
    bar = "█" * (c // 100)
    print(f"  {name:>10s}: {c:5d} {bar}")

# ── Class-Weighted Loss (critical for FER2013 imbalance) ─────
def compute_class_weights(labels, num_classes):
    cnt = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * cnt.get(i, 1)) for i in range(num_classes)]
    return torch.FloatTensor(weights)

class_weights = compute_class_weights(train_set.labels, NUM_CLASSES).to(DEVICE)
print(f"\n⚖️  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

# ── Model — EXACT match to backend EmotionNet ─────────────────
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
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
scaler    = torch.amp.GradScaler('cuda')

# ── Training ──────────────────────────────────────────────────
best_val_acc     = 0.0
patience_counter = 0
OUTPUT_PATH      = "/kaggle/working/emotion_model.pth"

print(f"\n{'='*70}")
print(f"🚀 Starting training: {NUM_EPOCHS} epochs, LR={LR}, BS={BATCH_SIZE}")
print(f"{'='*70}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_set)

    # ── Val ───────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct = 0.0, 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total   = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            val_loss    += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            for pred, lbl in zip(preds, labels):
                c = lbl.item()
                per_class_total[c] += 1
                if pred.item() == c:
                    per_class_correct[c] += 1

    val_loss /= len(val_loader)
    val_acc   = val_correct / len(val_set)
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:02d} | Train: {train_loss:.4f}/{train_acc:.4f} "
          f"| Val: {val_loss:.4f}/{val_acc:.4f} | LR: {lr:.6f}")

    # Per-class accuracy every 5 epochs
    if epoch % 5 == 0 or epoch == NUM_EPOCHS:
        print("  Per-class accuracy:")
        for i, name in enumerate(EMOTIONS):
            t = per_class_total[i]
            c = per_class_correct[i]
            a = c / t if t > 0 else 0
            print(f"    {name:>10s}: {a:.4f} ({c}/{t})")

    # ── Save best ─────────────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
            "val_acc":              val_acc,
        }, OUTPUT_PATH)
        print(f"  ★ New best! Val Acc: {val_acc:.4f} → Saved to {OUTPUT_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

print(f"\n{'='*70}")
print(f"🎉 Training complete! Best Val Accuracy: {best_val_acc:.4f}")
print(f"   Download: {OUTPUT_PATH}")
print(f"   Place in: PROJECT/data/weights/emotion_model.pth")
print(f"{'='*70}")
