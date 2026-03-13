"""
================================================================
 KAGGLE TRAINING NOTEBOOK — Age & Gender Prediction (UTKFace)
 v2 — Architecture matches backend AgeGenderModel exactly
================================================================
Instructions:
  1. Create a new Kaggle Notebook from UTKFace dataset page
  2. Enable GPU: Settings → Accelerator → GPU T4 x2
  3. Paste this entire script into a code cell and Run All

Output: 'age_gender_model.pth' in /kaggle/working/
Place in: PROJECT/data/weights/age_gender_model.pth
================================================================
"""

import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 30
LR         = 1e-4
PATIENCE   = 5
VAL_SPLIT  = 0.15

# ── Dataset ───────────────────────────────────────────────────
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []
        root    = Path(root_dir)
        pattern = re.compile(r"^(\d+)_(\d+)_")
        for img_path in root.rglob("*.jpg"):
            m = pattern.match(img_path.name)
            if m:
                age, gender = int(m.group(1)), int(m.group(2))
                if 0 <= age <= 116 and gender in [0, 1]:
                    self.samples.append((img_path, age, gender))
        print(f"Found {len(self.samples)} valid UTKFace samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, gender = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)

# ── Transforms & DataLoaders ──────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DATA_ROOT = "/kaggle/input/datasets/jangedoo/utkface-new/UTKFace"

dataset   = UTKFaceDataset(DATA_ROOT, transform=transform)
val_size  = int(VAL_SPLIT * len(dataset))
train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model — EXACT match to backend AgeGenderNet ───────────────
# Backend: backbone.fc = Identity(), then separate age_head and gender_head
class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()           # ← Must match backend exactly
        self.backbone    = backbone
        self.age_head    = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        features    = self.backbone(x)
        age_pred    = self.age_head(features).squeeze(1)
        gender_pred = self.gender_head(features)
        return age_pred, gender_pred

model            = AgeGenderNet().to(DEVICE)
age_criterion    = nn.L1Loss()
gender_criterion = nn.CrossEntropyLoss()
optimizer        = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler           = torch.amp.GradScaler('cuda')

# ── Training ──────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
OUTPUT_PATH      = "/kaggle/working/age_gender_model.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, ages, genders in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
        images, ages, genders = images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            age_pred, gen_pred = model(images)
            loss = age_criterion(age_pred, ages) + gender_criterion(gen_pred, genders)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    model.eval()
    val_loss, val_mae, val_gen_acc = 0.0, 0.0, 0
    with torch.no_grad():
        for images, ages, genders in val_loader:
            images, ages, genders = images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            with torch.amp.autocast('cuda'):
                age_pred, gen_pred = model(images)
                loss = age_criterion(age_pred, ages) + gender_criterion(gen_pred, genders)
            val_loss    += loss.item()
            val_mae     += age_criterion(age_pred, ages).item() * images.size(0)
            val_gen_acc += (gen_pred.argmax(dim=1) == genders).sum().item()

    val_loss    /= len(val_loader)
    val_mae     /= len(val_set)
    val_gen_acc /= len(val_set)
    print(f"Epoch {epoch:03d} | Loss: {running_loss/len(train_loader):.4f} "
          f"| Val Loss: {val_loss:.4f} | Age MAE: {val_mae:.2f} | Gender Acc: {val_gen_acc:.4f}")

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
print(f"   Place in: PROJECT/data/weights/age_gender_model.pth")
