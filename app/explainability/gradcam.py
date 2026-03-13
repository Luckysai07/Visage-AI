"""
Phase 5 — Grad-CAM Explainability Module.
Generates heatmaps showing which facial regions influenced model predictions.
Uses captum's LayerGradCam for clean integration with PyTorch models.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.core.device import DEVICE
from app.utils.image_utils import pil_to_tensor, denormalize_tensor
from app.utils.visualization import overlay_heatmap, pil_to_base64

logger = logging.getLogger(__name__)


def _get_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Automatically find the last Conv2d layer of a model.
    Works for ResNet-style architectures.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


class GradCAM:
    """
    Grad-CAM implementation using manual backward hooks.
    Compatible with any CNN that has Conv2d layers.

    Usage:
        cam = GradCAM(model, target_layer=model.layer4[1].conv2)
        heatmap = cam.generate(input_tensor, class_idx=3)
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: Optional[nn.Module] = None,
        device:       Optional[torch.device] = None,
    ):
        self.model        = model
        self.device       = device or DEVICE
        self.target_layer = target_layer or _get_last_conv_layer(model)

        if self.target_layer is None:
            raise ValueError("Could not find a Conv2d layer in the model.")

        self.gradients:  Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,   # (1, C, H, W)
        class_idx:    Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_idx:    Target class index. If None, uses predicted class.

        Returns:
            Heatmap as float32 ndarray in [0, 1], shape (H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # Backward pass for target class
        self.model.zero_grad()
        score = output[:, class_idx].sum()
        score.backward()

        # Compute Grad-CAM
        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam_np = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    def generate_overlay(
        self,
        face_image:   Image.Image,
        class_idx:    Optional[int] = None,
        alpha:        float = 0.45,
    ) -> Tuple[np.ndarray, str]:
        """
        Full Grad-CAM pipeline: process image → generate heatmap → overlay.

        Args:
            face_image: PIL RGB face image.
            class_idx:  Target class (None = predicted class).
            alpha:      Heatmap blend strength.

        Returns:
            (overlay_bgr: np.ndarray, heatmap_base64: str)
        """
        tensor = pil_to_tensor(face_image).to(self.device)
        heatmap = self.generate(tensor, class_idx=class_idx)

        # Convert face to BGR OpenCV for overlaying
        face_bgr = cv2.cvtColor(np.array(face_image.resize((224, 224))),
                                cv2.COLOR_RGB2BGR)
        overlay  = overlay_heatmap(face_bgr, heatmap, alpha=alpha)
        b64      = pil_to_base64(
            Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        )
        return overlay, b64


class FaceExplainer:
    """
    High-level explainability wrapper for common use cases.
    Generates Grad-CAM overlays for emotion and attribute models.
    """

    def __init__(
        self,
        emotion_model_nn:   Optional[nn.Module] = None,
        attribute_model_nn: Optional[nn.Module] = None,
        device:             Optional[torch.device] = None,
    ):
        self.device = device or DEVICE
        self.emotion_cam:   Optional[GradCAM] = None
        self.attribute_cam: Optional[GradCAM] = None

        if emotion_model_nn is not None:
            self.emotion_cam = GradCAM(emotion_model_nn, device=self.device)
        if attribute_model_nn is not None:
            self.attribute_cam = GradCAM(attribute_model_nn, device=self.device)

    def explain_emotion(
        self,
        face_image: Image.Image,
        emotion_class_idx: Optional[int] = None,
    ) -> Optional[str]:
        """Return base64 Grad-CAM heatmap for emotion prediction."""
        if self.emotion_cam is None:
            return None
        _, b64 = self.emotion_cam.generate_overlay(face_image, class_idx=emotion_class_idx)
        return b64

    def explain_attribute(
        self,
        face_image: Image.Image,
        attribute_idx: Optional[int] = None,
    ) -> Optional[str]:
        """Return base64 Grad-CAM heatmap for attribute prediction."""
        if self.attribute_cam is None:
            return None
        _, b64 = self.attribute_cam.generate_overlay(face_image, class_idx=attribute_idx)
        return b64
