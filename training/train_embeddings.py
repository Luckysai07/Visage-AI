"""
Training Script — Fine-tune InceptionResnetV1 for perfect identity separation.
Uses Triplet Loss to push same-identity embeddings together and others apart.

Expects a dataset where each subfolder is an identity.
Example structure:
    data/identities/
        user1/
            img1.jpg
            img2.jpg
        user2/
            img1.jpg
"""

import logging
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.models.embedding_model import EmbeddingModel
from app.utils.image_utils import face_preprocess_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16      # Adjusted for Triplet Mining
LEARNING_RATE = 1e-4
MARGIN        = 0.5     # Triplet Loss Margin
NUM_EPOCHS    = 20
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TripletFaceDataset(Dataset):
    """
    Dataset that returns triplets (Anchor, Positive, Negative).
    - Anchor:   An image of person A.
    - Positive: Another image of person A.
    - Negative: An image of person B.
    """
    def __init__(self, identities_dir: Path):
        self.identities_dir = identities_dir
        self.identity_to_images = {}
        
        for identity_folder in identities_dir.iterdir():
            if identity_folder.is_dir():
                images = list(identity_folder.glob("*"))
                if len(images) >= 2:
                    self.identity_to_images[identity_folder.name] = images
        
        self.identities = list(self.identity_to_images.keys())
        if len(self.identities) < 2:
            raise ValueError("Need at least 2 identities with 2 images each for Triplet Training.")

    def __len__(self) -> int:
        return sum(len(imgs) for imgs in self.identity_to_images.values())

    def __getitem__(self, idx: int):
        # Pick anchor identity
        anchor_id = random.choice(self.identities)
        
        # Pick Anchor and Positive (different images of same person)
        anc_path, pos_path = random.sample(self.identity_to_images[anchor_id], 2)
        
        # Pick Negative (image of different person)
        neg_id = random.choice([i for i in self.identities if i != anchor_id])
        neg_path = random.choice(self.identity_to_images[neg_id])
        
        # Load and preprocess
        anc_img = face_preprocess_transform(Image.open(anc_path).convert("RGB"))
        pos_img = face_preprocess_transform(Image.open(pos_path).convert("RGB"))
        neg_img = face_preprocess_transform(Image.open(neg_path).convert("RGB"))
        
        return anc_img, pos_img, neg_img


def train_embeddings(identities_path: Path):
    logger.info(f"Starting Embedding Fine-tuning (Triplet Loss) on {DEVICE}")
    
    dataset = TripletFaceDataset(identities_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    emb_model = EmbeddingModel(device=DEVICE)
    model = emb_model.model
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    
    best_loss = float("inf")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        for i, (anc, pos, neg) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad()
            
            anc_emb = model(anc)
            pos_emb = model(pos)
            neg_emb = model(neg)
            
            loss = criterion(anc_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss
            }
            torch.save(checkpoint, str(settings.EMBEDDING_WEIGHTS))
            logger.info(f"  → Saved new best weights to {settings.EMBEDDING_WEIGHTS}")

    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to identity dataset directory")
    args = parser.parse_args()
    
    train_embeddings(Path(args.data))
