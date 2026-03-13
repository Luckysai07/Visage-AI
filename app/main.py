"""
Phase 6 — FastAPI Application Entry Point.
Initializes the app, registers routes, manages model lifecycle.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router, set_pipeline, set_db_builder

# ─── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Application lifespan (startup / shutdown) ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup; clean up on shutdown."""
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Create required directories
    settings.create_directories()

    # Initialize pipeline (loads all models)
    logger.info("Loading inference pipeline…")
    from app.pipeline.face_pipeline import FacePipeline
    pipeline = FacePipeline()
    set_pipeline(pipeline)

    # Initialize database builder (shares models with pipeline)
    from app.pipeline.database_builder import DatabaseBuilder
    builder = DatabaseBuilder()
    set_db_builder(builder)

    logger.info("✓ System ready. Visit http://localhost:8000/docs for API documentation.")
    yield

    # Shutdown
    logger.info("Shutting down — releasing GPU memory…")
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## AI Face Analytics & Hybrid Image Retrieval System

A complete end-to-end face analysis pipeline:

- **Face Detection** — MTCNN with landmark extraction
- **Demographic Prediction** — Age, gender (ResNet18/UTKFace)  
- **Emotion Recognition** — 7-class classifier (ResNet18/FER2013)
- **Attribute Prediction** — 40 CelebA attributes
- **Deep Embeddings** — 512-dim ArcFace-style (InceptionResnetV1/VGGFace2)
- **Hybrid Retrieval** — Attribute filtering + FAISS cosine similarity
- **Explainability** — Grad-CAM heatmaps

### Quick Start
1. `POST /api/analyze` — Upload an image for full analysis
2. `POST /api/build-database` — Index your image collection
3. `POST /api/search` — Search by face attributes or image
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API routes ───────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api")

# ─── Static file serving ─────────────────────────────────────────────────────
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(settings.RESULTS_DIR)), name="results")

# ─── Root redirect ────────────────────────────────────────────────────────────
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        workers=1,       # Keep single worker; models are GPU-resident
    )
