"""
Phase 6 — Full Inference Pipeline Orchestrator.
Wires all models together for single-image end-to-end inference.
"""

import io
import uuid
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from app.core.config import settings
from app.core.device import DEVICE
from app.models.face_detector import FaceDetector
from app.models.rcnn_face_detector import RCNNFaceDetector
from app.models.age_gender_model import AgeGenderModel
from app.models.emotion_model import EmotionModel
from app.models.attribute_model import AttributeModel
from app.models.embedding_model import EmbeddingModel
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.attribute_filter import AttributeFilter
from app.retrieval.supabase_filter import SupabaseFilter
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
        1. Face detection — RCNN (Faster R-CNN) primary, MTCNN fallback
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
        self.detector      = FaceDetector(device=device)      # MTCNN (fallback)
        self.rcnn_detector = RCNNFaceDetector(device=device)  # Faster R-CNN (primary)
        self.age_gender    = AgeGenderModel(device=device)
        self.emotion       = EmotionModel(device=device)
        self.attribute     = AttributeModel(device=device)
        self.embedding     = EmbeddingModel(device=device)

        # ── Retrieval components ─────────────────────────────────────────────
        self.faiss_index = FaissIndex()
        
        if settings.USE_SUPABASE:
            self.attr_filter = SupabaseFilter()
            logger.info("Using Supabase as metadata store.")
        else:
            self.attr_filter = AttributeFilter()
            logger.info("Using local SQLite as metadata store.")

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
        store_in_db:        bool = False,
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
                        "search_score":      float,
                    }
                ],
                "detector_used":  str,
            }
        """
        top_k = top_k or settings.DEFAULT_TOP_K

        # ── Step 1: Face Detection ─────────────────────────────────────────
        # Use Faster R-CNN as primary detector; fall back to MTCNN if needed.
        if settings.RCNN_USE_AS_DETECTOR:
            detection = self.rcnn_detector.detect(image)
            if detection["count"] == 0:
                logger.info("RCNN found 0 faces — falling back to MTCNN.")
                detection = self.detector.detect(image)
                detector_used = "MTCNN (fallback)"
            else:
                detector_used = "Faster R-CNN"
        else:
            detection = self.detector.detect(image)
            detector_used = "MTCNN"
        face_count = detection["count"]
        logger.info(f"Detected {face_count} face(s) via {detector_used}.")

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

        for i, face_img_aligned in enumerate(detection["face_images"]):
            face_img_loose = detection["face_images_loose"][i]
            
            face_result: Dict[str, Any] = {
                "index":      i,
                "bbox":       detection["boxes"][i],
                "confidence": round(detection["confidences"][i], 4),
                "is_aligned": detection.get("is_aligned", [False])[i],
            }
            # ── Age & Gender (Use Loose Crop) ───────────────────────────────
            ag = self.age_gender.predict(face_img_loose)
            face_result.update({
                "age":               ag["age"],
                "gender":            ag["gender"],
                "gender_confidence": ag["gender_confidence"],
            })

            # ── Emotion (Use Aligned Crop) ──────────────────────────────────
            emo = self.emotion.predict(face_img_aligned)
            face_result.update({
                "emotion":             emo["emotion"],
                "emotion_confidence":  emo["confidence"],
                "emotion_all_scores":  emo["all_scores"],
            })

            # ── CelebA Attributes (Use Loose Crop) ────────────────────────────
            attr = self.attribute.predict(face_img_loose)
            face_result.update({
                "attributes":          attr["attributes"],
                "present_attributes":  attr["present_attributes"],
                "attribute_scores":    attr["attribute_scores"],
            })

            # ── Embedding + Hybrid Search (Use Aligned Crop) ─────────────────
            emb = self.embedding.extract(face_img_aligned)
            similar = []
            
            # IMPROVEMENT: Only allow search if the face is high-quality and aligned
            if not face_result["is_aligned"]:
                logger.warning(f"Face {i} is not perfectly aligned. Search results may be inaccurate.")
                face_result["search_warning"] = "Low quality alignment: results may be inaccurate."
            else:
                face_result["search_warning"] = None

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
            
            # Key metric for search "perfection": Top similarity score
            face_result["search_score"] = float(similar[0]["similarity"]) if similar else 0.0

            # ── Grad-CAM Explainability (Use appropriate crops) ───────────────
            if generate_heatmap:
                emotion_idx = settings.EMOTION_CLASSES.index(emo["emotion"]) \
                    if emo["emotion"] in settings.EMOTION_CLASSES else None
                face_result["heatmap_emotion"]    = self.explainer.explain_emotion(
                    face_img_aligned, emotion_class_idx=emotion_idx
                )
                face_result["heatmap_attribute"]  = self.explainer.explain_attribute(
                    face_img_loose
                )
            else:
                face_result["heatmap_emotion"]   = None
                face_result["heatmap_attribute"] = None

            # ── Database Storage (Optional) ──────────────────────────────────
            if store_in_db:
                try:
                    face_id = self._store_face(
                        face_img_aligned=face_img_aligned,
                        face_img_loose=face_img_loose,
                        ag_result=ag,
                        emo_result=emo,
                        attr_result=attr,
                        emb_vector=emb,
                        bbox=detection["boxes"][i],
                        detection_confidence=detection["confidences"][i]
                    )
                    face_result["db_id"] = face_id
                    logger.info(f"Stored uploaded face in DB (ID: {face_id})")
                except Exception as e:
                    logger.error(f"Failed to store uploaded face in DB: {e}")
                    face_result["db_id"] = None

            faces_output.append(face_result)

        if store_in_db and face_count > 0:
            self.faiss_index.save()

        return {
            "face_count":     face_count,
            "detected_image": detected_b64,
            "faces":          faces_output,
            "detector_used":  detector_used,
        }

    def _store_face(
        self,
        face_img_aligned:   Image.Image,
        face_img_loose:     Image.Image,
        ag_result:          Dict[str, Any],
        emo_result:         Dict[str, Any],
        attr_result:        Dict[str, Any],
        emb_vector:         np.ndarray,
        bbox:               List[float],
        detection_confidence: float,
    ) -> int:
        """Helper to store a single face in metadata DB and FAISS."""
        image_id  = str(uuid.uuid4())
        faiss_pos = self.faiss_index.size

        # Upload to Storage if enabled
        storage_url = None
        if settings.USE_SUPABASE:
            storage_url = self.attr_filter.upload_face_image(face_img_aligned, image_id)

        # Insert metadata
        face_db_id = self.attr_filter.insert_face(
            image_path="UPLOADED",
            image_id=image_id,
            age=ag_result["age"],
            gender=ag_result["gender"],
            gender_confidence=ag_result["gender_confidence"],
            emotion=emo_result["emotion"],
            emotion_confidence=emo_result["confidence"],
            attributes=attr_result["attributes"],
            bbox=tuple(bbox),
            detection_confidence=detection_confidence,
            faiss_position=faiss_pos,
            storage_url=storage_url,
        )

        # Add to FAISS
        self.faiss_index.add_single(emb_vector, face_db_id)
        
        return face_db_id
