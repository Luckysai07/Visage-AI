"""
================================================================
 KAGGLE TRAINING NOTEBOOK — 40 Facial Attributes (CelebA)
================================================================
Instructions:
  1. Create a new Kaggle Notebook at https://www.kaggle.com/code
  2. Add Dataset: "CelebFaces Attributes (CelebA) Dataset" 
     (search in Add Data panel)
  3. Enable GPU: Settings → Accelerator → GPU T4 x2
  4. Paste this entire script into a code cell and click Run All

Output: 'attribute_model.pth' saved in /kaggle/working/
Download it and place it in: PROJECT/data/weights/attribute_model.pth
================================================================
"""

# ──────────────────────────────────────────────────────────────
# CELL 1 — Imports
# ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# CELL 2 — Config
# ──────────────────────────────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 64
NUM_EPOCHS  = 20
LR          = 1e-4
PATIENCE    = 4
NUM_ATTRS   = 40

# ──────────────────────────────────────────────────────────────
# CELL 3 — Dataset
# ──────────────────────────────────────────────────────────────
class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, partition_file, split="train", transform=None):
        self.img_dir   = Path(img_dir)
        self.transform = transform

        # Parse attributes (1 / -1) → (1 / 0)
        attr_df    = pd.read_csv(attr_file, sep=r'\s+', header=1)
        part_df    = pd.read_csv(partition_file, sep=r'\s+', header=None, names=["image", "split"])
        split_map  = {"train": 0, "val": 1, "test": 2}

        mask       = part_df["split"] == split_map[split]
        valid_imgs = set(part_df[mask]["image"].values)

        self.samples = []
        self.attr_names = attr_df.columns.tolist()
        for img_name, row in attr_df.iterrows():
            if img_name in valid_imgs:
                labels = torch.tensor([(v + 1) // 2 for v in row.values], dtype=torch.float32)
                self.samples.append((img_name, labels))

        print(f"[{split}] Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, labels = self.samples[idx]
        img = Image.open(self.img_dir / img_name).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, labels

# ──────────────────────────────────────────────────────────────
# CELL 4 — Data Paths (Kaggle CelebA layout)
# ──────────────────────────────────────────────────────────────
BASE        = "/kaggle/input/celeba-dataset"
IMG_DIR     = f"{BASE}/img_align_celeba/img_align_celeba"
ATTR_FILE   = f"{BASE}/list_attr_celeba.csv"
PART_FILE   = f"{BASE}/list_eval_partition.csv"

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_set   = CelebADataset(IMG_DIR, ATTR_FILE, PART_FILE, "train", train_transform)
val_set     = CelebADataset(IMG_DIR, ATTR_FILE, PART_FILE, "val",   val_transform)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ──────────────────────────────────────────────────────────────
# CELL 5 — Model
# ──────────────────────────────────────────────────────────────
model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, NUM_ATTRS), nn.Sigmoid())
model    = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler    = torch.cuda.amp.GradScaler()

# ──────────────────────────────────────────────────────────────
# CELL 6 — Training
# ──────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
OUTPUT_PATH      = "/kaggle/working/attribute_model.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            val_loss += loss.item()
            preds     = (outputs > 0.5).float()
            val_acc  += (preds == labels).float().mean(dim=1).sum().item()

    val_loss /= len(val_loader)
    val_acc  /= len(val_set)
    print(f"Epoch {epoch:03d} | Train Loss: {running_loss/len(train_loader):.4f} "
          f"| Val Loss: {val_loss:.4f} | Val Avg Attr Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_PATH)
        print(f"  ✅ Saved best model → {OUTPUT_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

print(f"\n🎉 Training complete! Best Val Loss: {best_val_loss:.4f}")
