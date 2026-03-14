"""
Phase 3 — Image Database Builder.
Batch-processes a folder of images through the full pipeline:
   detect face → align → predict attributes → extract embedding → store in SQLite + FAISS
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from tqdm import tqdm

from app.core.config import settings
from app.models.face_detector import FaceDetector
from app.models.rcnn_face_detector import RCNNFaceDetector
from app.models.age_gender_model import AgeGenderModel
from app.models.emotion_model import EmotionModel
from app.models.attribute_model import AttributeModel
from app.models.embedding_model import EmbeddingModel
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.attribute_filter import AttributeFilter
from app.retrieval.supabase_filter import SupabaseFilter
from app.utils.image_utils import bytes_to_pil, load_image_pil

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatabaseBuilder:
    """
    Builds the face retrieval database from a directory of images.

    Processing flow per image:
    1. Load image
    2. Detect faces (MTCNN)
    3. For each detected face:
       a. Predict age + gender
       b. Predict emotion
       c. Predict CelebA attributes
       d. Extract 512-dim embedding
       e. Store metadata in SQLite
       f. Add embedding to FAISS index
    4. Save FAISS index to disk
    """

    def __init__(
        self,
        detector:    Optional[FaceDetector]    = None,
        rcnn_detector: Optional[RCNNFaceDetector] = None,
        age_gender:  Optional[AgeGenderModel]  = None,
        emotion:     Optional[EmotionModel]    = None,
        attribute:   Optional[AttributeModel]  = None,
        embedding:   Optional[EmbeddingModel]  = None,
        faiss_index: Optional[FaissIndex]      = None,
        db_filter:   Optional[Union[AttributeFilter, SupabaseFilter]] = None,
    ):
        """All model arguments are optional — created lazily if not provided."""
        self._detector      = detector
        self._rcnn_detector = rcnn_detector
        self._age_gender    = age_gender
        self._emotion    = emotion
        self._attribute  = attribute
        self._embedding  = embedding
        self._faiss      = faiss_index
        self._db         = db_filter

    # ── Lazy initialization ────────────────────────────────────────────────────
    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    @property
    def rcnn_detector(self) -> RCNNFaceDetector:
        if self._rcnn_detector is None:
            self._rcnn_detector = RCNNFaceDetector()
        return self._rcnn_detector

    @property
    def age_gender(self) -> AgeGenderModel:
        if self._age_gender is None:
            self._age_gender = AgeGenderModel()
        return self._age_gender

    @property
    def emotion(self) -> EmotionModel:
        if self._emotion is None:
            self._emotion = EmotionModel()
        return self._emotion

    @property
    def attribute(self) -> AttributeModel:
        if self._attribute is None:
            self._attribute = AttributeModel()
        return self._attribute

    @property
    def embedding(self) -> EmbeddingModel:
        if self._embedding is None:
            self._embedding = EmbeddingModel()
        return self._embedding

    @property
    def faiss_index(self) -> FaissIndex:
        if self._faiss is None:
            self._faiss = FaissIndex()
        return self._faiss

    @property
    def db(self) -> Union[AttributeFilter, SupabaseFilter]:
        if self._db is None:
            if settings.USE_SUPABASE:
                self._db = SupabaseFilter()
            else:
                self._db = AttributeFilter()
        return self._db

    # ── Core processing ────────────────────────────────────────────────────────

    def process_image(self, image_path: Path) -> int:
        """
        Process a single image file and add all detected faces to the database.

        Returns:
            Number of faces successfully added to the database.
        """
        try:
            pil_img = load_image_pil(image_path)
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            return 0

        # Use same detection logic as FacePipeline
        if settings.RCNN_USE_AS_DETECTOR:
            detection = self.rcnn_detector.detect(pil_img)
            if detection["count"] == 0:
                detection = self.detector.detect(pil_img)
        else:
            detection = self.detector.detect(pil_img)

        if detection["count"] == 0:
            return 0

        added = 0
        for i, face_img_aligned in enumerate(detection["face_images"]):
            face_img_loose = detection["face_images_loose"][i]
            try:
                # Aligned for search/emotion, Loose for demographics
                ag_result   = self.age_gender.predict(face_img_loose)
                emo_result  = self.emotion.predict(face_img_aligned)
                attr_result = self.attribute.predict(face_img_loose)
                emb_vector  = self.embedding.extract(face_img_aligned)

                # Assign FAISS position (current size before adding)
                faiss_pos = self.faiss_index.size
                image_id  = str(uuid.uuid4())

                # Upload to Cloud if using Supabase
                storage_url = None
                if isinstance(self.db, SupabaseFilter):
                    # Use aligned crop for UI
                    storage_url = self.db.upload_face_image(face_img_aligned, image_id)

                # Store in DB
                face_db_id = self.db.insert_face(
                    image_path=str(image_path),
                    image_id=image_id,
                    age=ag_result["age"],
                    gender=ag_result["gender"],
                    gender_confidence=ag_result["gender_confidence"],
                    emotion=emo_result["emotion"],
                    emotion_confidence=emo_result["confidence"],
                    attributes=attr_result["attributes"],
                    bbox=tuple(detection["boxes"][i]),
                    detection_confidence=detection["confidences"][i],
                    faiss_position=faiss_pos,
                    storage_url=storage_url,
                )

                # Add to FAISS using SQLite row ID for mapping
                self.faiss_index.add_single(emb_vector, face_db_id)
                added += 1

            except Exception as e:
                logger.warning(f"Failed to process face {i} in {image_path}: {e}")
                continue

        return added

    def build_from_directory(
        self,
        image_dir: Path,
        clear_existing: bool = False,
        max_images: Optional[int] = None,
    ) -> dict:
        """
        Scan a directory and build the database from all found images.

        Args:
            image_dir:      Path to directory containing images.
            clear_existing: If True, clear database and FAISS index before building.
            max_images:     Limit on number of images to process (for testing).

        Returns:
            Summary statistics dict.
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if clear_existing:
            self.db.clear()
            self.faiss_index.reset()
            logger.info("Cleared existing database.")

        image_files = [
            f for f in image_dir.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if max_images:
            image_files = image_files[:max_images]

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        total_images = 0
        total_faces  = 0
        failed       = 0

        for img_path in tqdm(image_files, desc="Building database"):
            faces_added = self.process_image(img_path)
            total_images += 1
            if faces_added > 0:
                total_faces += faces_added
            else:
                failed += 1

        # Save FAISS index to disk
        self.faiss_index.save()

        stats = {
            "images_processed": total_images,
            "faces_indexed":    total_faces,
            "images_no_face":   failed,
            "total_in_db":      self.db.count(),
        }
        logger.info(f"Database build complete: {stats}")
        return stats
