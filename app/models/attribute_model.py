"""
Phase 2 Level 2 — CelebA Multi-Label Facial Attribute Predictor.
ResNet18 backbone with 40 binary sigmoid outputs.
Attributes used for database filtering in hybrid retrieval.
"""

import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from app.core.config import settings
from app.core.device import DEVICE
from app.utils.image_utils import pil_to_tensor

logger = logging.getLogger(__name__)

# Threshold for classifying a sigmoid output as "True"
ATTRIBUTE_THRESHOLD = 0.5


class AttributeNet(nn.Module):
    """ResNet18 with 40 parallel binary sigmoid output heads for CelebA attributes."""

    def __init__(self, num_attributes: int = 40):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_attributes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))   # (B, 40) probabilities


class AttributeModel:
    """
    CelebA 40-attribute predictor.
    Outputs per-attribute probabilities used for:
    1. Hybrid retrieval filtering
    2. Database indexing
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.attribute_names: List[str] = settings.CELEBA_ATTRIBUTES
        self.model = AttributeNet(num_attributes=len(self.attribute_names)).to(self.device)
        self._load_weights()
        self.model.eval()
        logger.info(f"AttributeModel ready with {len(self.attribute_names)} attributes on {self.device}")

    def _load_weights(self):
        weights_path = settings.ATTRIBUTE_WEIGHTS
        if weights_path.exists():
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded attribute weights from {weights_path}")
        else:
            logger.warning(
                f"No trained attribute weights at {weights_path}. "
                "Run training/train_attributes.py to fine-tune on CelebA."
            )

    @torch.no_grad()
    def predict(self, face_image: Image.Image) -> Dict[str, Any]:
        """
        Predict all 40 CelebA attributes from a face image.

        Args:
            face_image: PIL RGB aligned face.

        Returns:
            {
                "attributes":        dict[str -> bool],
                "attribute_scores":  dict[str -> float],
                "present_attributes": list[str],
            }
        """
        tensor = pil_to_tensor(face_image).to(self.device)
        probs  = self.model(tensor)[0].cpu()   # (40,)

        attributes: Dict[str, bool]  = {}
        scores: Dict[str, float] = {}
        present: List[str] = []

        for name, prob in zip(self.attribute_names, probs):
            p = float(prob.item())
            scores[name]     = round(p, 4)
            attributes[name] = p >= ATTRIBUTE_THRESHOLD
            if attributes[name]:
                present.append(name)

        return {
            "attributes":         attributes,
            "attribute_scores":   scores,
            "present_attributes": present,
        }

    def predict_batch(self, face_images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Predict attributes for a batch of face images (more efficient)."""
        from app.utils.image_utils import preprocess_transform
        tensors = torch.stack([preprocess_transform(img) for img in face_images]).to(self.device)
        with torch.no_grad():
            batch_probs = self.model(tensors).cpu()   # (N, 40)

        results = []
        for probs in batch_probs:
            attributes, scores, present = {}, {}, []
            for name, prob in zip(self.attribute_names, probs):
                p = float(prob.item())
                scores[name] = round(p, 4)
                attributes[name] = p >= ATTRIBUTE_THRESHOLD
                if attributes[name]:
                    present.append(name)
            results.append({
                "attributes":         attributes,
                "attribute_scores":   scores,
                "present_attributes": present,
            })
        return results

    def get_model_for_gradcam(self) -> nn.Module:
        return self.model.model

    def save_weights(self, epoch: int, optimizer_state: dict = None):
        settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {"epoch": epoch, "model_state_dict": self.model.state_dict()}
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, str(settings.ATTRIBUTE_WEIGHTS))
        logger.info(f"Saved attribute checkpoint to {settings.ATTRIBUTE_WEIGHTS}")
