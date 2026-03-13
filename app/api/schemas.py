"""
Phase 6 — Pydantic API Schemas.
Request/response models for all FastAPI endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Sub-models ───────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class EmotionResult(BaseModel):
    emotion:     str = Field(description="Predicted emotion label")
    confidence:  float = Field(ge=0.0, le=1.0)
    all_scores:  Dict[str, float] = Field(description="Per-class probabilities")


class SimilarImage(BaseModel):
    face_db_id:   int
    image_id:     str
    image_path:   str
    similarity:   float = Field(description="Cosine similarity score, 0–1")
    age:          Optional[int]   = None
    gender:       Optional[str]   = None
    emotion:      Optional[str]   = None
    attributes:   Optional[Dict[str, Any]] = None


class FaceResult(BaseModel):
    index:                int
    bbox:                 List[float]  = Field(description="[x1, y1, x2, y2]")
    detection_confidence: float

    # Level 1 — Demographics
    age:                  Optional[int]   = None
    gender:               Optional[str]   = None
    gender_confidence:    Optional[float] = None

    # Level 1 — Emotion
    emotion:              Optional[str]   = None
    emotion_confidence:   Optional[float] = None
    emotion_all_scores:   Optional[Dict[str, float]] = None

    # Level 2 — Mid-level Attributes
    attributes:           Optional[Dict[str, bool]]  = None
    present_attributes:   Optional[List[str]]        = None
    attribute_scores:     Optional[Dict[str, float]] = None

    # Level 3 — Retrieval
    similar_images:       List[SimilarImage] = Field(default_factory=list)

    # Phase 5 — Explainability
    heatmap_emotion:      Optional[str] = Field(None, description="Base64 JPEG Grad-CAM heatmap")
    heatmap_attribute:    Optional[str] = Field(None, description="Base64 JPEG Grad-CAM heatmap")


# ─── Top-level response ────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    face_count:      int
    detected_image:  str = Field(description="Base64 JPEG with bounding boxes drawn")
    faces:           List[FaceResult]
    message:         Optional[str] = None


# ─── Search request ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    gender:           Optional[str]  = None
    emotion:          Optional[str]  = None
    age_min:          Optional[int]  = Field(None, ge=0, le=116)
    age_max:          Optional[int]  = Field(None, ge=0, le=116)
    attributes:       Optional[Dict[str, bool]] = Field(
        None,
        description="CelebA attribute constraints, e.g. {\"Smiling\": true}"
    )
    top_k:            int = Field(10, ge=1, le=100)
    generate_heatmap: bool = True


# ─── Database build ────────────────────────────────────────────────────────────

class BuildDatabaseRequest(BaseModel):
    image_dir:      str  = Field(description="Absolute path to image directory")
    clear_existing: bool = False
    max_images:     Optional[int] = Field(None, ge=1)


class BuildDatabaseResponse(BaseModel):
    images_processed: int
    faces_indexed:    int
    images_no_face:   int
    total_in_db:      int
    status:           str = "complete"


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    version:      str
    device:       str
    db_faces:     int
    faiss_vectors: int
    gpu_name:     Optional[str] = None
