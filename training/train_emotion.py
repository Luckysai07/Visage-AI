"""
Training script for Emotion prediction on FER2013.
Supports: mixed precision (AMP), early stopping, class-weighted loss,
learning-rate scheduling, and strong data augmentation.

Usage (local):
    python training/train_emotion.py --epochs 30 --batch-size 64 --lr 0.0001

Usage (Kaggle / Colab):
    !python training/train_emotion.py --data /kaggle/input/fer2013 --epochs 30 --batch-size 64
"""

import logging
import argparse
import os
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from app.core.config import settings
from app.core.device import DEVICE
from app.models.emotion_model import EmotionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_class_weights(dataset, num_classes: int):
    """Compute inverse-frequency class weights for imbalanced FER2013."""
    counts = Counter(dataset.labels)
    total = len(dataset)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    logger.info(f"Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.FloatTensor(weights)


def train():
    parser = argparse.ArgumentParser(description="Train Emotion Model on FER2013")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data", type=str, default=str(settings.FER2013_DIR),
                        help="Path to FER2013 root (containing train/ and test/ subdirs)")
    args = parser.parse_args()

    device = DEVICE
    data_dir = args.data
    logger.info(f"Device: {device} | Data: {data_dir}")

    # ── Data Augmentation ─────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    from training.datasets.fer2013 import FER2013Dataset

    train_set = FER2013Dataset(data_dir, split="train", transform=train_transform)
    val_set   = FER2013Dataset(data_dir, split="test",  transform=val_transform)

    if len(train_set) == 0:
        logger.error(f"No training images found at {data_dir}/train/")
        logger.error("Expected structure: train/<emotion_name>/*.jpg")
        logger.error(f"Emotion classes: {settings.EMOTION_CLASSES}")
        return

    logger.info(f"Train: {len(train_set)} images | Val: {len(val_set)} images")

    # Log class distribution
    train_counts = Counter(train_set.labels)
    for cls_name, cls_idx in train_set.class_to_idx.items():
        logger.info(f"  {cls_name}: {train_counts.get(cls_idx, 0)} images")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=min(settings.NUM_WORKERS, os.cpu_count() or 2), pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=settings.VAL_BATCH_SIZE, shuffle=False,
        num_workers=min(settings.NUM_WORKERS, os.cpu_count() or 2), pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_wrapper = EmotionModel(device=device)
    model = model_wrapper.model

    # ── Class-Weighted Loss ───────────────────────────────────────────────────
    class_weights = compute_class_weights(train_set, len(settings.EMOTION_CLASSES))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    use_amp = settings.MIXED_PRECISION and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    patience_counter = 0

    logger.info(f"Starting emotion training | LR: {args.lr} | Epochs: {args.epochs} | AMP: {use_amp}")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        # ── Training Loop ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_set)

        # ── Validation Loop ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        per_class_correct = [0] * len(settings.EMOTION_CLASSES)
        per_class_total   = [0] * len(settings.EMOTION_CLASSES)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

                # Per-class accuracy
                for pred, label in zip(preds, labels):
                    cls = label.item()
                    per_class_total[cls] += 1
                    if pred.item() == cls:
                        per_class_correct[cls] += 1

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_set)

        # Step the LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────────────
        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Per-class accuracy every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info("  Per-class accuracy:")
            for i, cls_name in enumerate(settings.EMOTION_CLASSES):
                total = per_class_total[i]
                correct = per_class_correct[i]
                acc = correct / total if total > 0 else 0
                logger.info(f"    {cls_name:>10s}: {acc:.4f} ({correct}/{total})")

        # ── Save Best Model ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_wrapper.save_weights(epoch, optimizer.state_dict())
            logger.info(f"  ★ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= settings.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {settings.PATIENCE} epochs)")
                break

    logger.info("=" * 70)
    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Weights saved to: {settings.EMOTION_WEIGHTS}")


if __name__ == "__main__":
    train()
