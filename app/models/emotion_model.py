"""
Phase 2 Level 1 — Emotion Prediction Model.
ResNet18 fine-tuned on FER2013 for 7-class emotion classification.
Classes: angry, disgust, fear, happy, sad, surprise, neutral.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from app.core.config import settings
from app.core.device import DEVICE
from app.utils.image_utils import pil_to_tensor

logger = logging.getLogger(__name__)


class EmotionNet(nn.Module):
    """ResNet18 backbone with a 7-class emotion output head."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # (B, 7) logits


class EmotionModel:
    """
    Emotion classifier with clean predict() interface.
    Falls back to ImageNet backbone if FER2013 weights unavailable.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.classes = settings.EMOTION_CLASSES
        self.model = EmotionNet(num_classes=len(self.classes)).to(self.device)
        self._load_weights()
        self.model.eval()
        logger.info(f"EmotionModel ready on {self.device}")

    def _load_weights(self):
        weights_path = settings.EMOTION_WEIGHTS
        if weights_path.exists():
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded emotion weights from {weights_path}")
        else:
            logger.warning(
                f"No trained emotion weights at {weights_path}. "
                "Run training/train_emotion.py to fine-tune on FER2013."
            )

    @torch.no_grad()
    def predict(self, face_image: Image.Image) -> Dict[str, Any]:
        """
        Predict emotion from a face image.

        Args:
            face_image: PIL RGB aligned face image.

        Returns:
            {
                "emotion":         str,
                "confidence":      float,
                "all_scores":      dict[emotion_name -> float],
            }
        """
        tensor = pil_to_tensor(face_image).to(self.device)
        logits = self.model(tensor)                       # (1, 7)
        probs  = torch.softmax(logits, dim=1)[0].cpu()   # (7,)

        best_idx   = int(probs.argmax().item())
        confidence = float(probs[best_idx].item())
        all_scores = {cls: round(float(p.item()), 4) for cls, p in zip(self.classes, probs)}

        return {
            "emotion":    self.classes[best_idx],
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def get_model_for_gradcam(self) -> nn.Module:
        """Return the underlying nn.Module for Grad-CAM."""
        return self.model.model   # The inner ResNet

    def save_weights(self, epoch: int, optimizer_state: dict = None):
        settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {"epoch": epoch, "model_state_dict": self.model.state_dict()}
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, str(settings.EMOTION_WEIGHTS))
        logger.info(f"Saved emotion checkpoint to {settings.EMOTION_WEIGHTS}")
