"""
Image utility functions — loading, preprocessing, and normalization.
Shared across all model modules and the pipeline.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
from torchvision import transforms

from app.core.config import settings


# ─── ImageNet normalization stats ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Standard preprocessing transform ─────────────────────────────────────────
preprocess_transform = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

face_preprocess_transform = transforms.Compose([
    transforms.Resize((160, 160)),   # FaceNet / InceptionResnetV1 expects 160x160
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_image_pil(path: Union[str, Path]) -> Image.Image:
    """Load an image as a PIL RGB image."""
    return Image.open(str(path)).convert("RGB")


def load_image_cv2(path: Union[str, Path]) -> np.ndarray:
    """Load an image as a NumPy BGR array (OpenCV convention)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw image bytes to PIL RGB image."""
    import io
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_tensor(pil_img: Image.Image, for_embedding: bool = False) -> torch.Tensor:
    """
    Convert PIL image to a normalized tensor.
    Args:
        pil_img: Input PIL RGB image.
        for_embedding: If True, use FaceNet normalization (-1..1) and 160x160.
                       If False, use ImageNet normalization and 224x224.
    Returns:
        Tensor of shape (1, C, H, W).
    """
    if for_embedding:
        tensor = face_preprocess_transform(pil_img)
    else:
        tensor = preprocess_transform(pil_img)
    return tensor.unsqueeze(0)  # Add batch dimension


def denormalize_tensor(tensor: torch.Tensor, for_embedding: bool = False) -> np.ndarray:
    """
    Reverse normalization on a (C, H, W) tensor → uint8 HWC numpy array.
    Useful for visualization.
    """
    tensor = tensor.clone().cpu()
    if for_embedding:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    else:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def resize_image(img: Union[np.ndarray, Image.Image],
                 size: Tuple[int, int]) -> Union[np.ndarray, Image.Image]:
    """Resize an image (handles both PIL and ndarray)."""
    if isinstance(img, Image.Image):
        return img.resize(size, Image.LANCZOS)
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def validate_image(pil_img: Image.Image) -> Tuple[bool, str]:
    """
    Check that an image is valid for processing.
    Returns (is_valid, error_message).
    """
    if pil_img.mode not in ("RGB", "L", "RGBA"):
        return False, f"Unsupported image mode: {pil_img.mode}"
    w, h = pil_img.size
    if w < 32 or h < 32:
        return False, f"Image too small: {w}x{h} (minimum 32x32)"
    return True, ""


def crop_face(image: Image.Image,
              bbox: Tuple[float, float, float, float],
              margin: int = 20) -> Image.Image:
    """
    Crop a face region from an image using a bounding box.
    Args:
        image: PIL RGB image.
        bbox: (x1, y1, x2, y2) bounding box.
        margin: Extra pixels to include around the bounding box.
    Returns:
        Cropped face as PIL RGB image.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    W, H = image.size
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)
    return image.crop((x1, y1, x2, y2))
