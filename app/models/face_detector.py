"""
Phase 1 — Face Detection Module.
Uses facenet-pytorch MTCNN for accurate multi-face detection,
landmark extraction, and aligned face cropping.
"""

import logging
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

import torch
from facenet_pytorch import MTCNN

from app.core.config import settings
from app.core.device import DEVICE

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    MTCNN-based face detector with alignment and cropping.

    Detects faces, returns:
    - Bounding boxes (x1, y1, x2, y2)
    - 5-point facial landmarks (eyes, nose, mouth corners)
    - Aligned & cropped face images (PIL RGB, 160x160 for embedding)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.mtcnn = MTCNN(
            image_size=160,               # Output crop size for embeddings
            margin=settings.MTCNN_MARGIN,
            min_face_size=settings.FACE_MIN_SIZE,
            thresholds=settings.MTCNN_THRESHOLDS,
            factor=0.709,
            post_process=True,           # Normalize pixel values in output
            keep_all=True,               # Detect ALL faces (not just largest)
            device=self.device,
        )
        # Separate MTCNN for getting raw boxes/landmarks without cropping
        self._mtcnn_detect = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=settings.FACE_MIN_SIZE,
            thresholds=settings.MTCNN_THRESHOLDS,
            post_process=False,
        )
        logger.info(f"FaceDetector initialized on {self.device}")

    def detect(
        self,
        image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Detect all faces in an image.

        Args:
            image: PIL RGB image.

        Returns:
            {
                "boxes":       List of [x1, y1, x2, y2],
                "landmarks":   List of (5, 2) arrays,
                "confidences": List of float,
                "faces":       List of (3, 160, 160) tensors (normalized),
                "face_images": List of PIL RGB face crops (160x160),
                "count":        int
            }
        """
        result: Dict[str, Any] = {
            "boxes": [],
            "landmarks": [],
            "confidences": [],
            "faces": [],
            "face_images": [],
            "face_images_loose": [],
            "count": 0,
        }

        # ── Step 1: Get bounding boxes and landmarks ───────────────────────
        boxes, probs, landmarks = self._mtcnn_detect.detect(image, landmarks=True)

        if boxes is None or len(boxes) == 0:
            logger.debug("No faces detected.")
            return result

        # ── Step 2: Filter by confidence ───────────────────────────────────
        valid_indices = [
            i for i, p in enumerate(probs)
            if p is not None and p >= settings.MTCNN_THRESHOLDS[-1]
        ]
        if not valid_indices:
            return result

        # ── Step 3: Crop aligned face tensors (for Search/Emotion) ────────
        face_tensors = self.mtcnn(image)
        if face_tensors is None:
            return result

        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)

        # ── Step 4: Crop loose faces (for Age/Gender) ────────────────────
        from app.utils.alignment import warp_face
        face_images = self._tensors_to_pil(face_tensors)
        face_images_loose = []
        for i, box in enumerate(boxes):
            # Create an Aligned Loose Crop with 40% margin
            lms = landmarks[i] if landmarks is not None else None
            if lms is not None:
                # Use our new warped alignment with 0.4 margin
                loose_pil = warp_face(
                    image, 
                    lms, 
                    target_size=settings.IMAGE_SIZE, 
                    margin=0.4
                )
            else:
                # Fallback to simple crop if landmarks missing
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                lx1, ly1 = max(0, x1 - w*0.4), max(0, y1 - h*0.4)
                lx2, ly2 = min(image.width, x2 + w*0.4), min(image.height, y2 + h*0.4)
                loose_pil = image.crop((lx1, ly1, lx2, ly2)).resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE), Image.LANCZOS)
                
            face_images_loose.append(loose_pil)

        # ── Step 5: Assemble results ───────────────────────────────────────
        for idx, orig_idx in enumerate(valid_indices):
            if idx >= len(face_tensors):
                break
            result["boxes"].append(boxes[orig_idx].tolist())
            result["landmarks"].append(
                landmarks[orig_idx] if landmarks is not None else []
            )
            result["confidences"].append(float(probs[orig_idx]))
            result["faces"].append(face_tensors[idx])              # Aligned tensor
            result["face_images"].append(face_images[idx])        # Aligned PIL
            result["face_images_loose"].append(face_images_loose[idx]) # Loose PIL
            # MTCNN align=True guarantees alignment if landmarks were found
            result.setdefault("is_aligned", []).append(True)

        result["count"] = len(result["boxes"])
        logger.debug(f"Detected {result['count']} face(s).")
        return result

    def detect_primary(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Detect the largest/most-confident face only.

        Returns:
            A single-face dict from detect() or None if no face found.
        """
        all_faces = self.detect(image)
        if all_faces["count"] == 0:
            return None

        # Pick most confident face
        best_idx = int(np.argmax(all_faces["confidences"]))
        return {
            "box":        all_faces["boxes"][best_idx],
            "landmark":   all_faces["landmarks"][best_idx],
            "confidence": all_faces["confidences"][best_idx],
            "face":       all_faces["faces"][best_idx],
            "face_image": all_faces["face_images"][best_idx],
            "face_image_loose": all_faces["face_images_loose"][best_idx],
        }

    @staticmethod
    def _tensors_to_pil(face_tensors: torch.Tensor) -> List[Image.Image]:
        """
        Convert (N, 3, 160, 160) MTCNN output tensor to list of PIL images.
        MTCNN post-processes to [-1, 1], so we un-normalize back.
        """
        pil_images = []
        for t in face_tensors:
            # Un-normalize from [-1, 1] → [0, 255]
            arr = ((t.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(arr))
        return pil_images
