"""
Phase 6 — FastAPI Route Definitions.
All API endpoints for the Face Analytics system.
"""

import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import (
    AnalyzeResponse,
    BuildDatabaseRequest,
    BuildDatabaseResponse,
    FaceResult,
    HealthResponse,
    SearchRequest,
    SimilarImage,
)
from app.core.config import settings
from app.core.device import get_device_info

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Dependency: shared pipeline singleton (set at startup) ───────────────────
_pipeline = None
_db_builder = None


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return _pipeline


def get_db_builder():
    if _db_builder is None:
        raise HTTPException(status_code=503, detail="DB builder not initialized.")
    return _db_builder


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


def set_db_builder(builder):
    global _db_builder
    _db_builder = builder


# ─── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — returns device info and database statistics."""
    pipeline = get_pipeline()
    device_info = get_device_info()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        device=device_info.get("device", "cpu"),
        db_faces=pipeline.attr_filter.count(),
        faiss_vectors=pipeline.faiss_index.size,
        gpu_name=device_info.get("gpu_name"),
    )


# ─── Main Analysis Endpoint ────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_image(
    file:             UploadFile = File(..., description="Image file to analyze"),
    top_k:            int        = Form(10,    description="Number of similar images to retrieve"),
    generate_heatmap: bool       = Form(True,  description="Generate Grad-CAM heatmaps"),
    filter_gender:    Optional[str] = Form(None, description="Filter retrieval by gender"),
    filter_emotion:   Optional[str] = Form(None, description="Filter retrieval by emotion"),
    filter_age_min:   Optional[int] = Form(None, description="Minimum age filter"),
    filter_age_max:   Optional[int] = Form(None, description="Maximum age filter"),
    pipeline = Depends(get_pipeline),
):
    """
    **Main endpoint** — upload an image and receive:
    - Face bounding boxes and confidence scores
    - Age, gender, emotion predictions
    - 40-attribute CelebA prediction
    - Top-K similar images from the database
    - Grad-CAM explainability heatmaps
    """
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Validate file size
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # Load image
    from app.utils.image_utils import bytes_to_pil, validate_image
    try:
        pil_image = bytes_to_pil(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    is_valid, err_msg = validate_image(pil_image)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    # Run pipeline
    try:
        result = pipeline.analyze(
            image=pil_image,
            top_k=top_k,
            generate_heatmap=generate_heatmap,
            filter_gender=filter_gender,
            filter_emotion=filter_emotion,
            filter_age_min=filter_age_min,
            filter_age_max=filter_age_max,
        )
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    # Convert raw dicts to Pydantic models
    faces_output = []
    for f in result.get("faces", []):
        similar = [SimilarImage(**s) for s in f.get("similar_images", [])]
        face_res = FaceResult(
            index=f["index"],
            bbox=f["bbox"],
            detection_confidence=f["confidence"],
            age=f.get("age"),
            gender=f.get("gender"),
            gender_confidence=f.get("gender_confidence"),
            emotion=f.get("emotion"),
            emotion_confidence=f.get("emotion_confidence"),
            emotion_all_scores=f.get("emotion_all_scores"),
            attributes=f.get("attributes"),
            present_attributes=f.get("present_attributes"),
            attribute_scores=f.get("attribute_scores"),
            similar_images=similar,
            heatmap_emotion=f.get("heatmap_emotion"),
            heatmap_attribute=f.get("heatmap_attribute"),
        )
        faces_output.append(face_res)

    return AnalyzeResponse(
        face_count=result["face_count"],
        detected_image=result["detected_image"],
        faces=faces_output,
        message=result.get("message"),
    )


# ─── Search Endpoint ──────────────────────────────────────────────────────────

@router.post("/search", tags=["Retrieval"])
async def search_by_attributes(
    request:  SearchRequest,
    file:     Optional[UploadFile] = File(None, description="Optional query face image"),
    pipeline  = Depends(get_pipeline),
):
    """
    Search the database by:
    - Attribute constraints (gender, emotion, age range, CelebA attributes)
    - Optionally combined with a query face image for embedding-based similarity
    """
    query_embedding = None

    if file is not None:
        content = await file.read()
        from app.utils.image_utils import bytes_to_pil
        pil_img = bytes_to_pil(content)
        detection = pipeline.detector.detect_primary(pil_img)
        if detection is not None:
            query_embedding = pipeline.embedding.extract(detection["face_image"])
        else:
            raise HTTPException(status_code=400, detail="No face detected in query image.")

    if query_embedding is not None:
        results = pipeline.searcher.search(
            query_embedding=query_embedding,
            k=request.top_k,
            gender=request.gender,
            emotion=request.emotion,
            age_min=request.age_min,
            age_max=request.age_max,
            attributes=request.attributes,
        )
    else:
        results = pipeline.searcher.search_by_attributes_only(
            gender=request.gender,
            emotion=request.emotion,
            age_min=request.age_min,
            age_max=request.age_max,
            attributes=request.attributes,
            limit=request.top_k,
        )

    return {"results": results, "count": len(results)}


# ─── Build Database Endpoint ──────────────────────────────────────────────────

@router.post("/build-database", response_model=BuildDatabaseResponse, tags=["Database"])
async def build_database(
    request:    BuildDatabaseRequest,
    background: BackgroundTasks,
    pipeline    = Depends(get_pipeline),
    builder     = Depends(get_db_builder),
):
    """
    Trigger database construction from a directory of images.
    Runs the full pipeline (detect → predict → embed) on each image.

    **Warning:** This is computationally intensive. Send as background task for large collections.
    """
    image_dir = Path(request.image_dir)
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.image_dir}")

    # Inject shared models into builder to avoid re-loading
    builder._detector   = pipeline.detector
    builder._age_gender = pipeline.age_gender
    builder._emotion    = pipeline.emotion
    builder._attribute  = pipeline.attribute
    builder._embedding  = pipeline.embedding
    builder._faiss      = pipeline.faiss_index
    builder._db         = pipeline.attr_filter

    try:
        stats = builder.build_from_directory(
            image_dir=image_dir,
            clear_existing=request.clear_existing,
            max_images=request.max_images,
        )
    except Exception as e:
        logger.exception("Database build error")
        raise HTTPException(status_code=500, detail=str(e))

    return BuildDatabaseResponse(
        images_processed=stats["images_processed"],
        faces_indexed=stats["faces_indexed"],
        images_no_face=stats["images_no_face"],
        total_in_db=stats["total_in_db"],
    )


# ─── Database Stats Endpoint ──────────────────────────────────────────────────

@router.get("/database/stats", tags=["Database"])
async def database_stats(pipeline=Depends(get_pipeline)):
    """Return current database statistics."""
    return {
        "total_faces":   pipeline.attr_filter.count(),
        "faiss_vectors": pipeline.faiss_index.size,
        "db_path":       str(settings.SQLITE_DB_PATH),
        "index_path":    str(settings.FAISS_INDEX_PATH),
    }
