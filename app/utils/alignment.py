"""
Face Alignment Utilities — Affine transforms for perfect search accuracy.
Aligns face images based on 5-point facial landmarks (eyes, nose, mouth corners).
Standardized alignment ensures consistent embedding extraction.
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


# Standard eye/mouth coordinates for 160x160 face crops (FaceNet / VGGFace2 convention)
# These coordinates represent where the landmarks should land in the aligned image.
STANDARD_FACE_160 = np.array([
    [30.2946, 51.6963],  # Left eye
    [65.5318, 51.5014],  # Right eye
    [48.0252, 71.7366],  # Nose
    [33.5493, 92.3655],  # Left mouth corner
    [62.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)

# Adjust for target shape scale (160 -> target_size)
def get_standard_landmarks(target_size: int = 160) -> np.ndarray:
    scale = target_size / 160.0
    return STANDARD_FACE_160 * scale


def warp_face(
    image: Union[Image.Image, np.ndarray],
    landmarks: np.ndarray,
    target_size: int = 160,
    margin: float = 0.1
) -> Image.Image:
    """
    Apply similarity transformation (affine) to align a face.

    Args:
        image: Original PIL or CV2 image.
        landmarks: (5, 2) numpy array of landmarks in face region.
        target_size: Output image width/height.
        margin: Padding around the face.

    Returns:
        Aligned and cropped PIL RGB image.
    """
    if isinstance(image, Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_cv = image

    # Landmarks are expected to be (5, 2)
    src_pts = landmarks.astype(np.float32)
    
    # Calculate target landmarks based on margin
    # margin=0 means tight crop, margin=0.5 means very loose
    dst_pts = get_standard_landmarks(target_size)
    
    if margin > 0:
        # Shift and scale landmarks to "zoom out"
        # We want the face (landmarks) to occupy a smaller portion of the target image
        center = target_size / 2.0
        scale = 1.0 / (1.0 + margin * 2)
        dst_pts = (dst_pts - center) * scale + center

    # Compute similarity transform (rotation + scaling + translation)
    # We use estimateAffinePartial2D which finds the best similarity transform.
    affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if affine_matrix is None:
        # Fallback to simple crop if transform estimation fails
        logger.warning("Affine transformation matrix estimation failed. Falling back to simple resize.")
        if isinstance(image, Image.Image):
            return image.resize((target_size, target_size), Image.BILINEAR)
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)).resize((target_size, target_size))

    # Perform the warp
    aligned_face = cv2.warpAffine(
        image_cv,
        affine_matrix,
        (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))


def get_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def check_alignment_quality(landmarks: np.ndarray) -> float:
    """
    Estimate alignment quality based on landmark geometric ratios.
    Returns a score 0..1.
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0

    # Basic check: eye distance should be roughly proportional to nose-mouth distance
    eye_dist = get_euclidean_distance(landmarks[0], landmarks[1])
    nose_mouth_dist = get_euclidean_distance(landmarks[2], (landmarks[3] + landmarks[4]) / 2)

    if eye_dist == 0: return 0.0
    ratio = nose_mouth_dist / eye_dist

    # Ideal human face ratio is around 0.6 - 1.2
    if 0.4 < ratio < 1.4:
        return 1.0
    return 0.5
