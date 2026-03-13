"""
Phase 3 — SQLite Database + Attribute Filtering Engine.
Stores face metadata: embeddings, age, gender, emotion, CelebA attributes, image paths.
Provides efficient SQL-based filtering for hybrid retrieval.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS faces (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path          TEXT    NOT NULL,
    image_id            TEXT    UNIQUE,
    -- Demographics
    age                 INTEGER,
    gender              TEXT,
    gender_confidence   REAL,
    -- Emotion
    emotion             TEXT,
    emotion_confidence  REAL,
    -- CelebA attributes stored as JSON blob
    attributes          TEXT,
    -- Bounding box of detected face
    bbox_x1             REAL,
    bbox_y1             REAL,
    bbox_x2             REAL,
    bbox_y2             REAL,
    detection_confidence REAL,
    -- Embedding stored in FAISS; we just keep the FAISS position here
    faiss_position      INTEGER,
    -- Timestamps
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_age    ON faces(age);
CREATE INDEX IF NOT EXISTS idx_gender ON faces(gender);
CREATE INDEX IF NOT EXISTS idx_emotion ON faces(emotion);
"""


class AttributeFilter:
    """
    SQLite-backed face metadata store and attribute filter.

    Responsibilities:
    1. Persist face records (demographics + CelebA attributes + FAISS position)
    2. Query records by attribute constraints for hybrid retrieval
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(CREATE_TABLE_SQL)
        logger.info(f"SQLite database ready at {self.db_path}")

    # ── Insertion ──────────────────────────────────────────────────────────────

    def insert_face(
        self,
        image_path: str,
        image_id: str,
        age: Optional[int],
        gender: Optional[str],
        gender_confidence: Optional[float],
        emotion: Optional[str],
        emotion_confidence: Optional[float],
        attributes: Optional[Dict[str, Any]],
        bbox: Optional[Tuple[float, float, float, float]],
        detection_confidence: Optional[float],
        faiss_position: int,
    ) -> int:
        """Insert a face record and return its SQLite row ID."""
        attrs_json = json.dumps(attributes) if attributes else None
        bbox_vals  = bbox if bbox else (None, None, None, None)

        sql = """
        INSERT OR REPLACE INTO faces
            (image_path, image_id, age, gender, gender_confidence,
             emotion, emotion_confidence, attributes,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             detection_confidence, faiss_position)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._get_conn() as conn:
            cursor = conn.execute(sql, (
                image_path, image_id,
                age, gender, gender_confidence,
                emotion, emotion_confidence, attrs_json,
                *bbox_vals,
                detection_confidence, faiss_position,
            ))
            return cursor.lastrowid

    # ── Filtering ─────────────────────────────────────────────────────────────

    def filter(
        self,
        gender:         Optional[str]  = None,
        emotion:        Optional[str]  = None,
        age_min:        Optional[int]  = None,
        age_max:        Optional[int]  = None,
        attributes:     Optional[Dict[str, bool]] = None,
        limit:          int = 1000,
    ) -> List[int]:
        """
        Filter database records by attribute constraints.

        Args:
            gender:     "male" | "female" | None (no filter)
            emotion:    e.g. "happy" | None
            age_min:    Minimum age
            age_max:    Maximum age
            attributes: Dict of CelebA attribute name → required bool value
                        e.g. {"Smiling": True, "Eyeglasses": False}
            limit:      Max number of candidate IDs to return

        Returns:
            List of SQLite row IDs (face.id) matching all constraints.
        """
        conditions: List[str] = []
        params:     List[Any] = []

        if gender is not None:
            conditions.append("LOWER(gender) = LOWER(?)")
            params.append(gender)

        if emotion is not None:
            conditions.append("LOWER(emotion) = LOWER(?)")
            params.append(emotion)

        if age_min is not None:
            conditions.append("age >= ?")
            params.append(age_min)

        if age_max is not None:
            conditions.append("age <= ?")
            params.append(age_max)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT id, attributes FROM faces {where_clause} LIMIT {limit}"

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        # Post-filter on JSON attribute blob (efficient for CelebA attributes)
        if not attributes:
            return [row["id"] for row in rows]

        filtered_ids: List[int] = []
        for row in rows:
            if row["attributes"] is None:
                continue
            try:
                stored_attrs = json.loads(row["attributes"])
            except (json.JSONDecodeError, TypeError):
                continue
            if all(
                stored_attrs.get(attr_name) == required_val
                for attr_name, required_val in attributes.items()
            ):
                filtered_ids.append(row["id"])

        return filtered_ids

    def get_by_id(self, face_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a full face record by SQLite ID."""
        sql = "SELECT * FROM faces WHERE id = ?"
        with self._get_conn() as conn:
            row = conn.execute(sql, (face_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        if record.get("attributes"):
            record["attributes"] = json.loads(record["attributes"])
        return record

    def get_by_ids(self, face_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve multiple face records by IDs."""
        if not face_ids:
            return []
        placeholders = ",".join("?" * len(face_ids))
        sql = f"SELECT * FROM faces WHERE id IN ({placeholders})"
        with self._get_conn() as conn:
            rows = conn.execute(sql, face_ids).fetchall()
        results = []
        for row in rows:
            rec = dict(row)
            if rec.get("attributes"):
                rec["attributes"] = json.loads(rec["attributes"])
            results.append(rec)
        return results

    def count(self) -> int:
        """Return total number of indexed faces."""
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def clear(self):
        """Remove all records from the database."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM faces")
        logger.info("Database cleared.")
