"""
Training script — Fine-tune Faster R-CNN on face detection data.

This script fine-tunes a Faster R-CNN (ResNet-50 FPN) with a 2-class head
(background + face) on labelled face datasets (e.g. WIDER FACE).

If no dataset annotations are available, a small synthetic dataset is
generated to demonstrate the training pipeline structure.

Usage:
    cd PROJECT_ROOT
    python training/train_rcnn_face.py

Outputs:
    data/weights/rcnn_face_model.pth
"""

import logging
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

# ─── Bootstrap project path ───────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES   = 2       # background (0) + face (1)
FACE_LABEL    = 1
BATCH_SIZE    = 2       # keep small for CPU train; increase for GPU
NUM_EPOCHS    = 10
LEARNING_RATE = 5e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FaceDetectionDataset(Dataset):
    """
    Dataset for Faster R-CNN face detection training.

    Expects a list of (image_path, boxes) tuples where boxes is a list of
    [x1, y1, x2, y2] face bounding boxes in pixel coordinates.

    Falls back to a synthetic dataset if no real data is provided.
    """

    def __init__(self, samples: List[Tuple[str, List[List[float]]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path, boxes = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(image)  # (3, H, W), float32 in [0,1]

        if len(boxes) == 0:
            # Background image — no faces
            target = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,),    dtype=torch.int64),
            }
        else:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.full((len(boxes),), FACE_LABEL, dtype=torch.int64)
            target   = {"boxes": boxes_t, "labels": labels_t}

        return img_tensor, target


def collate_fn(batch):
    """Custom collate to handle variable-size target dicts."""
    return tuple(zip(*batch))


# ─── Synthetic Dataset Helpers ────────────────────────────────────────────────

def _make_synthetic_face_image(
    width: int = 400,
    height: int = 400,
    num_faces: int = 1,
    save_path: str = None,
) -> Tuple[str, List[List[float]]]:
    """
    Create a synthetic image with coloured rectangles as face proxies.
    For demonstration / CI purposes only.
    """
    img   = Image.new("RGB", (width, height), color=(200, 200, 200))
    draw  = ImageDraw.Draw(img)
    boxes = []

    for _ in range(num_faces):
        bw = random.randint(40, 100)
        bh = random.randint(40, 100)
        x1 = random.randint(0, width  - bw)
        y1 = random.randint(0, height - bh)
        x2, y2 = x1 + bw, y1 + bh
        color = (random.randint(100,220), random.randint(80,180), random.randint(60,140))
        draw.ellipse([x1, y1, x2, y2], fill=color)
        boxes.append([float(x1), float(y1), float(x2), float(y2)])

    if save_path:
        img.save(save_path)
    return save_path, boxes


def _build_synthetic_dataset(
    n: int = 50,
    out_dir: Path = None,
) -> List[Tuple[str, List[List[float]]]]:
    """Generate n synthetic face-detection samples."""
    out_dir = out_dir or (settings.RAW_DATA_DIR / "synthetic_faces")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    logger.info(f"Generating {n} synthetic training samples in {out_dir} ...")
    for i in range(n):
        num_faces = random.randint(1, 3)
        path      = str(out_dir / f"face_{i:04d}.jpg")
        _, boxes  = _make_synthetic_face_image(save_path=path, num_faces=num_faces)
        samples.append((path, boxes))
    logger.info("Synthetic dataset ready.")
    return samples


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_rcnn_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    Build Faster R-CNN with ResNet-50 FPN backbone.
    Replaces the default COCO head with a custom num_classes head.
    """
    model     = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_feats  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(
    samples: List[Tuple[str, List[List[float]]]],
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float       = LEARNING_RATE,
):
    logger.info(f"Training on {DEVICE}  |  {len(samples)} samples  |  {num_epochs} epochs")

    # ── Split 80/20 ──────────────────────────────────────────────────────────
    random.shuffle(samples)
    split   = max(1, int(0.8 * len(samples)))
    train_s = samples[:split]
    val_s   = samples[split:] or samples[:1]  # always have at least 1 val sample

    train_ds = FaceDetectionDataset(train_s)
    val_ds   = FaceDetectionDataset(val_s)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=1,          shuffle=False, collate_fn=collate_fn
    )

    # ── Model & Optimizer ────────────────────────────────────────────────────
    model     = build_rcnn_model().to(DEVICE)
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ── Train epoch ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        avg_train = epoch_loss / max(len(train_loader), 1)

        # ── Validation epoch ─────────────────────────────────────────────────
        model.train()   # Faster R-CNN computes losses only in train mode
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = [img.to(DEVICE) for img in imgs]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()
        avg_val = val_loss / max(len(val_loader), 1)

        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d}/{num_epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
        )

        # ── Save best ────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_loss":         best_val_loss,
            }
            torch.save(checkpoint, str(settings.RCNN_WEIGHTS))
            logger.info(f"  → Best model saved to {settings.RCNN_WEIGHTS}")

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    return model


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Try to load real WIDER FACE annotations ──────────────────────────────
    wider_dir = settings.RAW_DATA_DIR / "WIDER_FACE"
    if wider_dir.exists():
        logger.info(f"Found WIDER FACE directory at {wider_dir}")
        logger.warning(
            "Automatic WIDER FACE parsing is not yet implemented. "
            "Falling back to synthetic data. "
            "Add a custom parse_wider_face() function here to use your dataset."
        )
        samples = _build_synthetic_dataset(n=100)
    else:
        logger.info(
            "WIDER FACE dataset not found. "
            "Using synthetic face dataset for demonstration."
        )
        samples = _build_synthetic_dataset(n=100)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(samples, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
