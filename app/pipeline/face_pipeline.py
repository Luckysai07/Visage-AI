"""
Phase 6 — Full Inference Pipeline Orchestrator.
Wires all models together for single-image end-to-end inference.
"""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from app.core.config import settings
from app.core.device import DEVICE
from app.models.face_detector import FaceDetector
from app.models.age_gender_model import AgeGenderModel
from app.models.emotion_model import EmotionModel
from app.models.attribute_model import AttributeModel
from app.models.embedding_model import EmbeddingModel
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.attribute_filter import AttributeFilter
from app.retrieval.hybrid_search import HybridSearch
from app.explainability.gradcam import FaceExplainer
from app.utils.visualization import draw_face_detections, pil_to_base64
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FacePipeline:
    """
    End-to-end face analytics pipeline.

    On initialization, loads all models once (using lazy singletons).
    On each call to analyze(), processes an image through:
        1. Face detection (MTCNN)
        2. For each face:
            a. Age + Gender prediction
            b. Emotion prediction
            c. Attribute prediction (CelebA 40)
            d. Embedding extraction (512-dim)
            e. Hybrid similarity search
            f. Grad-CAM explainability heatmap
    """

    def __init__(self):
        logger.info("Initializing FacePipeline — loading all models...")
        device = DEVICE

        # ── Models ──────────────────────────────────────────────────────────
        self.detector    = FaceDetector(device=device)
        self.age_gender  = AgeGenderModel(device=device)
        self.emotion     = EmotionModel(device=device)
        self.attribute   = AttributeModel(device=device)
        self.embedding   = EmbeddingModel(device=device)

        # ── Retrieval components ─────────────────────────────────────────────
        self.faiss_index = FaissIndex()
        self.attr_filter = AttributeFilter()
        db_loaded = self.faiss_index.load()
        if not db_loaded:
            logger.warning(
                "FAISS index empty — retrieval disabled until database is built. "
                "Call POST /api/build-database to index your image collection."
            )

        self.searcher = HybridSearch(
            faiss_index=self.faiss_index,
            attribute_filter=self.attr_filter,
        )

        # ── Explainability ───────────────────────────────────────────────────
        self.explainer = FaceExplainer(
            emotion_model_nn=self.emotion.get_model_for_gradcam(),
            attribute_model_nn=self.attribute.get_model_for_gradcam(),
            device=device,
        )

        logger.info("FacePipeline ready.")

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze(
        self,
        image:              Image.Image,
        top_k:              int  = None,
        generate_heatmap:   bool = True,
        # Retrieval filters
        filter_gender:      Optional[str]  = None,
        filter_emotion:     Optional[str]  = None,
        filter_age_min:     Optional[int]  = None,
        filter_age_max:     Optional[int]  = None,
        filter_attributes:  Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run full analysis on a single PIL image.

        Returns:
            {
                "face_count":     int,
                "detected_image": str (base64 JPEG with bounding boxes drawn),
                "faces": [
                    {
                        "index":           int,
                        "bbox":            [x1,y1,x2,y2],
                        "confidence":      float,
                        "age":             int,
                        "gender":          str,
                        "gender_confidence": float,
                        "emotion":         str,
                        "emotion_confidence": float,
                        "emotion_all_scores": dict,
                        "attributes":      dict,
                        "present_attributes": list[str],
                        "similar_images":  list[dict],
                        "heatmap_emotion": str (base64) | None,
                        "heatmap_attribute": str (base64) | None,
                    }
                ]
            }
        """
        top_k = top_k or settings.DEFAULT_TOP_K

        # ── Step 1: Face Detection ──────────────────────────────────────────
        detection = self.detector.detect(image)
        face_count = detection["count"]
        logger.info(f"Detected {face_count} face(s).")

        # Draw detection results on original image
        image_cv2 = pil_to_cv2(image)
        if face_count > 0:
            annotated = draw_face_detections(
                image=image_cv2,
                boxes=detection["boxes"],
                landmarks=detection["landmarks"],
                confidences=detection["confidences"],
            )
        else:
            annotated = image_cv2.copy()
        detected_b64 = pil_to_base64(cv2_to_pil(annotated))

        if face_count == 0:
            return {
                "face_count":     0,
                "detected_image": detected_b64,
                "faces":          [],
                "message":        "No faces detected in the uploaded image.",
            }

        # ── Step 2: Per-face analysis ───────────────────────────────────────
        faces_output: List[Dict[str, Any]] = []

        for i, face_img in enumerate(detection["face_images"]):
            face_result: Dict[str, Any] = {
                "index":      i,
                "bbox":       detection["boxes"][i],
                "confidence": round(detection["confidences"][i], 4),
            }

            # ── Age & Gender ────────────────────────────────────────────────
            ag = self.age_gender.predict(face_img)
            face_result.update({
                "age":               ag["age"],
                "gender":            ag["gender"],
                "gender_confidence": ag["gender_confidence"],
            })

            # ── Emotion ──────────────────────────────────────────────────────
            emo = self.emotion.predict(face_img)
            face_result.update({
                "emotion":             emo["emotion"],
                "emotion_confidence":  emo["confidence"],
                "emotion_all_scores":  emo["all_scores"],
            })

            # ── CelebA Attributes ─────────────────────────────────────────────
            attr = self.attribute.predict(face_img)
            face_result.update({
                "attributes":          attr["attributes"],
                "present_attributes":  attr["present_attributes"],
                "attribute_scores":    attr["attribute_scores"],
            })

            # ── Embedding + Hybrid Search ─────────────────────────────────────
            emb = self.embedding.extract(face_img)
            similar = []
            if self.faiss_index.size > 0:
                similar = self.searcher.search(
                    query_embedding=emb,
                    k=top_k,
                    gender=filter_gender,
                    emotion=filter_emotion,
                    age_min=filter_age_min,
                    age_max=filter_age_max,
                    attributes=filter_attributes,
                )
            face_result["similar_images"] = similar

            # ── Grad-CAM Explainability ───────────────────────────────────────
            if generate_heatmap:
                emotion_idx = settings.EMOTION_CLASSES.index(emo["emotion"]) \
                    if emo["emotion"] in settings.EMOTION_CLASSES else None
                face_result["heatmap_emotion"]    = self.explainer.explain_emotion(
                    face_img, emotion_class_idx=emotion_idx
                )
                face_result["heatmap_attribute"]  = self.explainer.explain_attribute(
                    face_img
                )
            else:
                face_result["heatmap_emotion"]   = None
                face_result["heatmap_attribute"] = None

            faces_output.append(face_result)

        return {
            "face_count":     face_count,
            "detected_image": detected_b64,
            "faces":          faces_output,
        }
