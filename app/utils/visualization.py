"""
Visualization utilities — drawing bounding boxes, landmarks,
and heatmap overlays on images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict, Any
import base64
import io


# ─── Color Palette ────────────────────────────────────────────────────────────
BBOX_COLOR     = (0, 255, 120)   # Neon green  (BGR for OpenCV)
LANDMARK_COLOR = (0, 180, 255)   # Orange
TEXT_BG_COLOR  = (0, 255, 120)
TEXT_COLOR     = (0, 0, 0)
HEATMAP_ALPHA  = 0.45


def draw_face_detections(
    image: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    landmarks: Optional[List[np.ndarray]] = None,
    confidences: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and landmarks on an OpenCV BGR image.
    Args:
        image:       OpenCV BGR image.
        boxes:       List of (x1, y1, x2, y2) bounding boxes.
        landmarks:   List of (5, 2) landmark arrays.
        confidences: Detection confidence scores.
        labels:      Optional text labels for each face.
    Returns:
        Annotated BGR image.
    """
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        # Build label
        label_parts = []
        if confidences is not None:
            label_parts.append(f"{confidences[i]:.2f}")
        if labels is not None and i < len(labels):
            label_parts.append(labels[i])
        label_text = " | ".join(label_parts) if label_parts else f"Face {i+1}"

        # Draw label background
        (lw, lh), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), TEXT_BG_COLOR, -1)
        cv2.putText(img, label_text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Draw landmarks
        if landmarks is not None and i < len(landmarks):
            for pt in landmarks[i]:
                cx, cy = int(pt[0]), int(pt[1])
                cv2.circle(img, (cx, cy), 3, LANDMARK_COLOR, -1)

    return img


def overlay_heatmap(
    face_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a face image.
    Args:
        face_image: BGR face image (H, W, 3) uint8.
        heatmap:    Grayscale heatmap (H, W) float32 in [0, 1].
        alpha:      Blend ratio for the heatmap.
        colormap:   OpenCV colormap ID.
    Returns:
        Blended BGR image (H, W, 3) uint8.
    """
    # Resize heatmap to match image
    h, w = face_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Normalize heatmap to 0–255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend
    blended = cv2.addWeighted(heatmap_colored, alpha, face_image, 1 - alpha, 0)
    return blended


def annotate_attributes(
    image: np.ndarray,
    attributes: Dict[str, Any],
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """
    Annotate an image with predicted attribute text.
    Args:
        image:      BGR image.
        attributes: Dict of attribute name → value.
        position:   Top-left corner for annotation block.
    Returns:
        Annotated BGR image.
    """
    img = image.copy()
    x, y = position
    line_height = 20
    for key, val in attributes.items():
        if isinstance(val, float):
            text = f"{key}: {val:.2f}"
        else:
            text = f"{key}: {val}"
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height
    return img


def image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR OpenCV image as a base64 string for API responses."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("Failed to encode image to base64.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 string for API responses."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_results_grid(
    images: List[np.ndarray],
    scores: Optional[List[float]] = None,
    cols: int = 5,
    thumb_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Create a grid thumbnail of retrieved similar images.
    Args:
        images:     List of BGR images.
        scores:     Optional similarity scores.
        cols:       Number of columns in the grid.
        thumb_size: (width, height) thumbnail size.
    Returns:
        Grid image as BGR ndarray.
    """
    thumbs = []
    for i, img in enumerate(images):
        thumb = cv2.resize(img, thumb_size)
        if scores is not None and i < len(scores):
            label = f"{scores[i]:.3f}"
            cv2.putText(thumb, label, (5, thumb_size[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1)
        thumbs.append(thumb)

    # Pad to fill grid
    rows_needed = (len(thumbs) + cols - 1) // cols
    while len(thumbs) < rows_needed * cols:
        thumbs.append(np.zeros((*thumb_size[::-1], 3), dtype=np.uint8))

    rows = []
    for r in range(rows_needed):
        row = np.hstack(thumbs[r * cols: (r + 1) * cols])
        rows.append(row)
    return np.vstack(rows)
