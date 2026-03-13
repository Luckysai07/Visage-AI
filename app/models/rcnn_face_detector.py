import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
from PIL import Image
from facenet_pytorch import MTCNN

from app.core.config import settings
from app.core.device import DEVICE
from app.utils.alignment import warp_face

logger = logging.getLogger(__name__)

# COCO label id for "person" — used when COCO pretrained weights are active
_COCO_PERSON_LABEL = 1

# Face crop size to match MTCNN output (keeps downstream models unchanged)
_FACE_CROP_SIZE = 160


class RCNNFaceDetector:
    """
    Two-Stage Face Detector:
    1. Faster R-CNN (ResNet-50 FPN) proposes bounding boxes.
    2. MTCNN extracts 5-point facial landmarks from the proposed boxes.
    3. Affine Warp aligns the face based on landmarks.

    This hybrid approach combines RCNN's robustness with MTCNN's precise alignment.
    """

    NUM_FACE_CLASSES = 2  # background (0) + face (1)

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.confidence_threshold = settings.RCNN_CONFIDENCE_THRESHOLD
        self._fine_tuned = False

        # Stage 1: RCNN for boxes
        self.model = self._build_model()
        self.model.eval()

        # Stage 2: MTCNN for landmarks only
        self.landmark_extractor = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False
        )

        logger.info(f"RCNNFaceDetector (Two-Stage) initialized on {self.device}")

    # ── Model Construction ────────────────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        """Build and return the Faster R-CNN model, loading weights."""
        weights_path = settings.RCNN_WEIGHTS

        if weights_path.exists():
            logger.info(f"Loading fine-tuned RCNN face weights from {weights_path}")
            model = fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.NUM_FACE_CLASSES
            )
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state)
            self._fine_tuned = True
        else:
            logger.warning("Using COCO pretrained Faster R-CNN for boxes.")
            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )

        return model.to(self.device)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect and ALIGN faces.

        Flow: RCNN Boxes -> MTCNN Landmarks -> Affine Warp Alignment.
        """
        result: Dict[str, Any] = {
            "boxes":       [],
            "landmarks":   [],
            "confidences": [],
            "faces":       [],
            "face_images": [],
            "face_images_loose": [],
            "count":       0,
        }

        # ── Stage 1: RCNN Boxes ──────────────────────────────────────────────
        img_tensor = TF.to_tensor(image.convert("RGB")).to(self.device)
        outputs = self.model([img_tensor])
        pred = outputs[0]

        boxes  = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        # Filter by class (person=1 in COCO, face=1 in fine-tuned)
        keep = [
            i for i in range(len(scores))
            if scores[i] >= self.confidence_threshold
            and labels[i] == 1
        ]

        if not keep:
            return result

        # ── Stage 2: Landmarks & Alignment ───────────────────────────────────
        img_w, img_h = image.size
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            
            # Implementation of "Box Smoothing": 
            # Slightly enlarge the box to ensure MTCNN has context to find landmarks.
            box_w, box_h = x2 - x1, y2 - y1
            x1, y1 = max(0, x1 - box_w * 0.1), max(0, y1 - box_h * 0.1)
            x2, y2 = min(img_w, x2 + box_w * 0.1), min(img_h, y2 + box_h * 0.1)
            
            box = [x1, y1, x2, y2]
            
            # Detect landmarks inside the RCNN proposed box
            _, _, lms = self.landmark_extractor.detect(image, landmarks=True)
            # Find the landmarks that belong to this box
            matched_lms = None
            if lms is not None:
                for face_lms in lms:
                    # Check if the nose (landmark index 2) is inside the box
                    nose = face_lms[2]
                    if x1 <= nose[0] <= x2 and y1 <= nose[1] <= y2:
                        matched_lms = face_lms
                        break
            
            if matched_lms is not None:
                # Use MTCNN's native extraction logic for "Perfect" accuracy 
                aligned_faces = self.landmark_extractor.extract(image, np.array([box]), save_path=None)
                if aligned_faces is not None and len(aligned_faces) > 0:
                    crop_tensor = aligned_faces[0] # Aligned and normalized to [-1, 1]
                    aligned_pil = self._tensor_to_pil(crop_tensor)
                    landmarks_out = matched_lms.tolist()
                    is_aligned = True
                else:
                    aligned_pil = image.crop(box).resize((_FACE_CROP_SIZE, _FACE_CROP_SIZE), Image.BILINEAR)
                    crop_tensor = self._pil_to_normalized_tensor(aligned_pil)
                    landmarks_out = []
                    is_aligned = False
            else:
                aligned_pil = image.crop(box).resize((_FACE_CROP_SIZE, _FACE_CROP_SIZE), Image.BILINEAR)
                crop_tensor = self._pil_to_normalized_tensor(aligned_pil)
                landmarks_out = []
                is_aligned = False

            # ── Aligned Loose Crop for Age/Gender (Box + 40% margin) ──
            from app.utils.alignment import warp_face
            if matched_lms is not None:
                loose_pil = warp_face(
                    image, 
                    matched_lms, 
                    target_size=settings.IMAGE_SIZE, 
                    margin=0.4
                )
            else:
                # Fallback to simple crop if landmarks missing
                loose_pil = image.crop((x1 - box_w*0.4, y1 - box_h*0.4, x2 + box_w*0.4, y2 + box_h*0.4)).resize(
                    (settings.IMAGE_SIZE, settings.IMAGE_SIZE), Image.LANCZOS
                )

            result["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
            result["landmarks"].append(landmarks_out)
            result["confidences"].append(float(scores[idx]))
            result["faces"].append(crop_tensor)
            result["face_images"].append(aligned_pil)
            result["face_images_loose"].append(loose_pil)
            result.setdefault("is_aligned", []).append(is_aligned)

        result["count"] = len(result["boxes"])
        return result

    def detect_primary(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        all_faces = self.detect(image)
        if all_faces["count"] == 0: return None
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
    def _pil_to_normalized_tensor(pil_img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(pil_img)
        t = (t - 0.5) / 0.5
        return t

    @staticmethod
    def _tensor_to_pil(face_tensor: torch.Tensor) -> Image.Image:
        """Un-normalize from [-1, 1] → [0, 255] PIL image."""
        arr = ((face_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def save_weights(self, epoch: int, optimizer_state: dict = None):
        settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {"epoch": epoch, "model_state_dict": self.model.state_dict()}
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, str(settings.RCNN_WEIGHTS))
