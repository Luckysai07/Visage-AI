"""
Training script for Multi-Label Attribute prediction on CelebA.
Supports: mixed precision (AMP), early stopping, learning-rate scheduling,
positive-case weighting, and strong data augmentation.
"""

import logging
import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from app.core.config import settings
from app.core.device import DEVICE
from app.models.attribute_model import AttributeModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train():
    parser = argparse.ArgumentParser(description="Train Attribute Model on CelebA")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data", type=str, default=str(settings.CELEBA_DIR),
                        help="Path to CelebA root directory")
    args = parser.parse_args()

    device = DEVICE
    data_dir = args.data
    logger.info(f"Device: {device} | Data: {data_dir}")

    # ── Data Augmentation ─────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    from training.datasets.celeba import CelebADataset
    train_set = CelebADataset(data_dir, split="train", transform=train_transform)
    val_set   = CelebADataset(data_dir, split="val",   transform=val_transform)

    if len(train_set) == 0:
        logger.error(f"No CelebA training images found at {data_dir}")
        return

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=min(settings.NUM_WORKERS, os.cpu_count() or 2), pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=settings.VAL_BATCH_SIZE, shuffle=False,
        num_workers=min(settings.NUM_WORKERS, os.cpu_count() or 2), pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_wrapper = AttributeModel(device=device)
    model = model_wrapper.model

    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    # Using BCEWithLogitsLoss is generally more stable than BCELoss(sigmoid)
    # But since AttributeNet currently includes sigmoid, we stick to BCELoss
    # or better, modify AttributeNet to return logits for training.
    criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    use_amp = settings.MIXED_PRECISION and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting CelebA training | Train: {len(train_set)} | Val: {len(val_set)}")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        # ── Training Loop ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validation Loop ───────────────────────────────────────────────
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_acc += (preds == labels).float().mean(dim=1).sum().item()

        val_loss /= len(val_loader)
        val_acc  /= len(val_set)
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | Avg Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_wrapper.save_weights(epoch, optimizer.state_dict())
            logger.info(f"  ★ New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= settings.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info("=" * 70)
    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
