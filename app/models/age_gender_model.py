"""
Phase 2 Level 1 — Age & Gender Prediction Model.
ResNet18 with two output heads:
  - Age regression (single scalar, 0–116 years)
  - Gender classification (2 classes: male/female)
Pretrained on ImageNet; fine-tuned on UTKFace.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from app.core.config import settings
from app.core.device import DEVICE
from app.utils.image_utils import pil_to_tensor

logger = logging.getLogger(__name__)

GENDER_CLASSES = ["male", "female"]


class AgeGenderNet(nn.Module):
    """
    ResNet18 with two parallel heads:
    - age_head: Linear(512 → 1)
    - gender_head: Linear(512 → 2)
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove original FC
        self.backbone = backbone
        self.age_head    = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.backbone(x)
        age_pred    = self.age_head(features).squeeze(1)      # (B,)
        gender_pred = self.gender_head(features)               # (B, 2)
        return age_pred, gender_pred


class AgeGenderModel:
    """
    Wrapper for AgeGenderNet providing a clean predict() interface.
    Loads pretrained weights from disk if available, otherwise uses
    ImageNet-pretrained backbone (suitable for inference with reduced accuracy
    until fine-tuned on UTKFace).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.model = AgeGenderNet().to(self.device)
        self._load_weights()
        self.model.eval()
        logger.info(f"AgeGenderModel ready on {self.device}")

    def _load_weights(self):
        weights_path = settings.AGE_GENDER_WEIGHTS
        if weights_path.exists():
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded age/gender weights from {weights_path}")
        else:
            logger.warning(
                f"No trained weights found at {weights_path}. "
                "Using ImageNet-pretrained backbone. "
                "Run training/train_age_gender.py to fine-tune."
            )

    @torch.no_grad()
    def predict(self, face_image: Image.Image, aligned_face: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Predict age and gender from a face image.
        Uses Dual-Crop strategy if aligned_face is provided:
        1. Predict on loose crop (face_image) - better for context
        2. Predict on tight aligned crop (aligned_face) - better for features
        3. Average the results for stability.

        Args:
            face_image: PIL RGB image of a loose face crop (standard).
            aligned_face: Optional PIL RGB image of a tight aligned face crop.

        Returns:
            {
                "age":            int,
                "gender":         "male" | "female",
                "gender_confidence": float (0–1),
                "age_raw":        float
            }
        """
        # --- Crop 1: Loose (Current default) ---
        tensor_loose = pil_to_tensor(face_image).to(self.device)
        age_raw_loose, gender_logits_loose = self.model(tensor_loose)
        
        if aligned_face is not None:
            # --- Crop 2: Aligned (Tight) ---
            tensor_aligned = pil_to_tensor(aligned_face).to(self.device)
            age_raw_aligned, gender_logits_aligned = self.model(tensor_aligned)
            
            # --- Average Age ---
            age_val = (float(age_raw_loose.cpu().item()) + float(age_raw_aligned.cpu().item())) / 2.0
            
            # --- Average Gender Probabilities ---
            probs_loose = torch.softmax(gender_logits_loose, dim=1)
            probs_aligned = torch.softmax(gender_logits_aligned, dim=1)
            gender_probs = ((probs_loose + probs_aligned) / 2.0)[0].cpu()
        else:
            age_val = float(age_raw_loose.cpu().item())
            gender_probs = torch.softmax(gender_logits_loose, dim=1)[0].cpu()

        age_val = max(0.0, min(116.0, age_val))   # Clamp to valid age range

        gender_idx   = int(gender_probs.argmax().item())
        gender_conf  = float(gender_probs[gender_idx].item())

        return {
            "age":                round(age_val),
            "gender":             GENDER_CLASSES[gender_idx],
            "gender_confidence":  round(gender_conf, 4),
            "age_raw":            round(age_val, 2),
        }

    def save_weights(self, epoch: int, optimizer_state: dict = None):
        """Save model checkpoint."""
        settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
        }
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, str(settings.AGE_GENDER_WEIGHTS))
        logger.info(f"Saved age/gender checkpoint to {settings.AGE_GENDER_WEIGHTS}")
