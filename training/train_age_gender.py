"""
Training script for Age & Gender prediction on UTKFace.
Supports mixed precision training (AMP) and early stopping.
"""

import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from app.core.config import settings
from app.core.device import DEVICE
from app.models.age_gender_model import AgeGenderModel
from training.datasets.utkface import UTKFaceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settings.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE)
    args = parser.parse_args()

    device = DEVICE
    
    # Data Augmentation & Loading
    transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = UTKFaceDataset(settings.UTKFACE_DIR, transform=transform)
    if len(dataset) == 0:
        logger.error(f"No UTKFace images found at {settings.UTKFACE_DIR}")
        return

    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # Overwrite transform for validation set (hacky but works for random_split)
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=settings.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=settings.VAL_BATCH_SIZE, shuffle=False,
                            num_workers=settings.NUM_WORKERS, pin_memory=True)

    # Model Setup
    model_wrapper = AgeGenderModel(device=device)
    model = model_wrapper.model
    
    age_criterion = nn.L1Loss()
    gender_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=settings.MIXED_PRECISION and device.type == "cuda")

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting training on {device} | Train: {train_size} | Val: {val_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for images, ages, genders in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            ages = ages.to(device).float()
            genders = genders.to(device).long()
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=settings.MIXED_PRECISION and device.type == "cuda"):
                age_preds, gender_preds = model(images)
                loss_age = age_criterion(age_preds, ages)
                loss_gender = gender_criterion(gender_preds, genders)
                # Weight age and gender losses equally
                loss = loss_age + loss_gender

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, val_age_mae, val_gender_acc = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, ages, genders in val_loader:
                images, ages, genders = images.to(device), ages.to(device).float(), genders.to(device).long()
                
                with torch.cuda.amp.autocast(enabled=settings.MIXED_PRECISION and device.type == "cuda"):
                    age_preds, gender_preds = model(images)
                    l_age = age_criterion(age_preds, ages)
                    l_gen = gender_criterion(gender_preds, genders)
                    loss = l_age + l_gen
                    
                val_loss += loss.item()
                val_age_mae += l_age.item() * images.size(0)
                val_gender_acc += (gender_preds.argmax(dim=1) == genders).sum().item()
                
        val_loss /= len(val_loader)
        val_age_mae /= len(val_set)
        val_gender_acc /= len(val_set)
        
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} "
                    f"| Val Loss: {val_loss:.4f} | Val Age MAE: {val_age_mae:.2f} "
                    f"| Val Gender Acc: {val_gender_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_wrapper.save_weights(epoch, optimizer.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= settings.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    train()
