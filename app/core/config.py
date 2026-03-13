"""
Central configuration for the AI Face Analytics System.
All paths, hyperparameters, and system settings live here.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


# ─── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "AI Face Analytics & Hybrid Image Retrieval System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Image Processing ───────────────────────────────────────────────────────
    IMAGE_SIZE: int = 224          # Standard input size for all models
    FACE_MIN_SIZE: int = 20        # Minimum face size for MTCNN detection
    MTCNN_THRESHOLDS: List[float] = [0.6, 0.7, 0.7]  # MTCNN detection confidence thresholds
    MTCNN_MARGIN: int = 20         # Margin around detected face bbox in pixels

    # ── Model Weights ──────────────────────────────────────────────────────────
    WEIGHTS_DIR: Path = BASE_DIR / "data" / "weights"
    AGE_GENDER_WEIGHTS: Path = WEIGHTS_DIR / "age_gender_model.pth"
    EMOTION_WEIGHTS: Path = WEIGHTS_DIR / "emotion_model.pth"
    ATTRIBUTE_WEIGHTS: Path = WEIGHTS_DIR / "attribute_model.pth"

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_DIR: Path = BASE_DIR / "data" / "database"
    SQLITE_DB_PATH: Path = DATABASE_DIR / "face_database.db"
    FAISS_INDEX_PATH: Path = DATABASE_DIR / "face_index.faiss"
    FAISS_ID_MAP_PATH: Path = DATABASE_DIR / "faiss_id_map.npy"

    # ── Embeddings ─────────────────────────────────────────────────────────────
    EMBEDDING_DIM: int = 512       # InceptionResnetV1 output dimension
    FAISS_METRIC: str = "cosine"   # "cosine" (IndexFlatIP on L2-normalized vecs)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 10        # Default number of similar images to return
    RETRIEVAL_MAX_CANDIDATES: int = 1000  # Max candidates after attribute filter

    # ── API ────────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "data" / "results"
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # ── Emotion Classes ────────────────────────────────────────────────────────
    EMOTION_CLASSES: List[str] = [
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    ]

    # ── CelebA Attribute Names ────────────────────────────────────────────────
    CELEBA_ATTRIBUTES: List[str] = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
        "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
        "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ]

    # ── Training ───────────────────────────────────────────────────────────────
    TRAIN_BATCH_SIZE: int = 16
    VAL_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 30
    PATIENCE: int = 7              # Early stopping patience
    MIXED_PRECISION: bool = True   # Use AMP for reduced GPU memory
    NUM_WORKERS: int = 4

    # ── Data Directories ──────────────────────────────────────────────────────
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    UTKFACE_DIR: Path = RAW_DATA_DIR / "UTKFace"
    FER2013_DIR: Path = RAW_DATA_DIR / "FER2013"
    CELEBA_DIR: Path = RAW_DATA_DIR / "CelebA"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def create_directories(self):
        """Create all required directories if they don't exist."""
        dirs = [
            self.WEIGHTS_DIR,
            self.DATABASE_DIR,
            self.UPLOAD_DIR,
            self.RESULTS_DIR,
            self.RAW_DATA_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
