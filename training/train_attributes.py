"""
Training script for Multi-Label Attribute prediction on CelebA.
Supports mixed precision training (AMP) and early stopping.
"""

import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from app.core.config import settings
from app.core.device import DEVICE
from app.models.attribute_model import AttributeModel
from training.datasets.celeba import CelebADataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settings.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE)
    args = parser.parse_args()

    device = DEVICE
    
    transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = CelebADataset(settings.CELEBA_DIR, split="train", transform=transform)
    val_set = CelebADataset(settings.CELEBA_DIR, split="val", transform=val_transform)
    
    if len(train_set) == 0:
        logger.error(f"No CelebA training images found. Ensure {settings.CELEBA_DIR} contains img_align_celeba/ etc.")
        return

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=settings.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=settings.VAL_BATCH_SIZE, shuffle=False,
                            num_workers=settings.NUM_WORKERS, pin_memory=True)

    model_wrapper = AttributeModel(device=device)
    model = model_wrapper.model
    
    # Binary Cross Entropy with Logits Loss since AttributeNet uses sigmoid? 
    # Wait, AttributeNet model currently applies torch.sigmoid(...)
    # Technically BCELoss works on probabilities.
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=settings.MIXED_PRECISION and device.type == "cuda")

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting CelebA training on {device} | Train: {len(train_set)} | Val: {len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=settings.MIXED_PRECISION and device.type == "cuda"):
                outputs = model(images)  # Already sigmoid probabilities
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                
                with torch.cuda.amp.autocast(enabled=settings.MIXED_PRECISION and device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_acc += (preds == labels).float().mean(dim=1).sum().item()
                
        val_loss /= len(val_loader)
        val_acc /= len(val_set)
        
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} "
                    f"| Val Loss: {val_loss:.4f} | Val Avg Acc: {val_acc:.4f}")

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
