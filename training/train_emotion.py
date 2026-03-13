"""
Training script for Emotion prediction on FER2013.
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
from app.models.emotion_model import EmotionModel
from training.datasets.fer2013 import FER2013Dataset

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = FER2013Dataset(settings.FER2013_DIR, split="train", transform=transform)
    val_set = FER2013Dataset(settings.FER2013_DIR, split="test", transform=val_transform)
    
    if len(train_set) == 0:
        logger.error(f"No FER2013 training images found at {settings.FER2013_DIR}/train")
        return

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=settings.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=settings.VAL_BATCH_SIZE, shuffle=False,
                            num_workers=settings.NUM_WORKERS, pin_memory=True)

    model_wrapper = EmotionModel(device=device)
    model = model_wrapper.model
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=settings.MIXED_PRECISION and device.type == "cuda")

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting generic emotion training on {device} | Train: {len(train_set)} | Val: {len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=settings.MIXED_PRECISION and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                
                with torch.cuda.amp.autocast(enabled=settings.MIXED_PRECISION and device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                val_acc += (outputs.argmax(dim=1) == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc /= len(val_set)
        
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} "
                    f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

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
