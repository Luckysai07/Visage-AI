"""
Training script for Age & Gender prediction on UTKFace.
Supports: mixed precision (AMP), early stopping, class-weighted loss (gender),
learning-rate scheduling, and strong data augmentation.
"""

import logging
import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from app.core.config import settings
from app.core.device import DEVICE
from app.models.age_gender_model import AgeGenderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train():
    parser = argparse.ArgumentParser(description="Train Age & Gender Model on UTKFace")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data", type=str, default=str(settings.UTKFACE_DIR),
                        help="Path to UTKFace dataset directory")
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    from training.datasets.utkface import UTKFaceDataset
    dataset = UTKFaceDataset(data_dir, transform=train_transform)

    if len(dataset) == 0:
        logger.error(f"No UTKFace images found at {data_dir}")
        return

    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Use a custom subset to ensure validation uses val_transform
    # (Simplified approach: swap transform during val loop is cleaner if single worker)
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=min(settings.NUM_WORKERS, os.cpu_count() or 2), pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=settings.VAL_BATCH_SIZE, shuffle=False,
        num_workers=min(settings.NUM_WORKERS, os.cpu_count() or 2), pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_wrapper = AgeGenderModel(device=device)
    model = model_wrapper.model

    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    age_criterion = nn.L1Loss()
    gender_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    use_amp = settings.MIXED_PRECISION and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting age/gender training | Train: {train_size} | Val: {val_size}")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        # ── Training Loop ─────────────────────────────────────────────────
        model.train()
        dataset.transform = train_transform # Ensure train transform
        
        train_loss = 0.0
        for images, ages, genders in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            ages = ages.to(device).float()
            genders = genders.to(device).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                age_preds, gender_preds = model(images)
                loss_age = age_criterion(age_preds, ages)
                loss_gender = gender_criterion(gender_preds, genders)
                loss = loss_age + (1.5 * loss_gender) # Slight weight to gender accuracy

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validation Loop ───────────────────────────────────────────────
        model.eval()
        dataset.transform = val_transform # Temporarily swap to val transform
        
        val_loss, val_age_mae, val_gender_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, ages, genders in val_loader:
                images, ages, genders = images.to(device), ages.to(device).float(), genders.to(device).long()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    age_preds, gender_preds = model(images)
                    l_age = age_criterion(age_preds, ages)
                    l_gen = gender_criterion(gender_preds, genders)
                    loss = l_age + l_gen
                    
                val_loss += loss.item()
                val_age_mae += l_age.item() * images.size(0)
                val_gender_acc += (gender_preds.argmax(dim=1) == genders).sum().item()

        val_loss /= len(val_loader)
        val_age_mae /= val_size
        val_gender_acc /= val_size
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | MAE: {val_age_mae:.2f} | Acc: {val_gender_acc:.4f} | "
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
