"""
Phase 4 — Hybrid Image Retrieval Engine.
Combines attribute-based SQL filtering and FAISS vector similarity search.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.attribute_filter import AttributeFilter

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid retrieval pipeline:
      Step 1 → Filter candidates from SQLite by attribute constraints
      Step 2 → Run FAISS similarity search restricted to those candidates
      Step 3 → Enrich results with full metadata from SQLite
    """

    def __init__(
        self,
        faiss_index: FaissIndex,
        attribute_filter: AttributeFilter,
    ):
        self.faiss_index      = faiss_index
        self.attribute_filter = attribute_filter

    def search(
        self,
        query_embedding: "np.ndarray",      # (512,) float32 L2-normalized
        k:               int = None,
        # ── Attribute filters (all optional) ──
        gender:          Optional[str]  = None,
        emotion:         Optional[str]  = None,
        age_min:         Optional[int]  = None,
        age_max:         Optional[int]  = None,
        attributes:      Optional[Dict[str, bool]] = None,
        use_filter:      bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-K most similar faces with optional attribute pre-filtering.

        Args:
            query_embedding: L2-normalized 512-dim face embedding.
            k:               Number of results (default from config).
            gender:          Filter by gender ("male" | "female").
            emotion:         Filter by emotion ("happy", "neutral", ...).
            age_min/max:     Age range filter.
            attributes:      CelebA attribute constraints dict.
            use_filter:      If False, skips attribute filtering (pure vector search).

        Returns:
            List of result dicts sorted by descending similarity:
            [{
                "image_id":        str,
                "image_path":      str,
                "similarity":      float,
                "age":             int,
                "gender":          str,
                "emotion":         str,
                "attributes":      dict,
            }, ...]
        """
        import numpy as np

        k = k or settings.DEFAULT_TOP_K

        # ── Step 1: Attribute filtering ────────────────────────────────────
        candidate_ids: Optional[List[int]] = None
        if use_filter and any(v is not None for v in [gender, emotion, age_min, age_max, attributes]):
            candidate_ids = self.attribute_filter.filter(
                gender=gender,
                emotion=emotion,
                age_min=age_min,
                age_max=age_max,
                attributes=attributes,
                limit=settings.RETRIEVAL_MAX_CANDIDATES,
            )
            if not candidate_ids:
                logger.info("Attribute filter returned 0 candidates — no results.")
                return []
            logger.info(f"Attribute filter narrowed to {len(candidate_ids)} candidates.")

        # ── Step 2: FAISS vector similarity search ─────────────────────────
        raw_results = self.faiss_index.search(
            query=query_embedding,
            k=k,
            candidate_ids=candidate_ids,
        )

        if not raw_results:
            return []

        # ── Step 3: Enrich with SQLite metadata ────────────────────────────
        result_ids = [img_id for img_id, _ in raw_results]
        score_map  = {img_id: score for img_id, score in raw_results}
        records    = self.attribute_filter.get_by_ids(result_ids)

        # Build output, keep original similarity ordering
        record_by_id = {r["id"]: r for r in records}
        output: List[Dict[str, Any]] = []

        for face_id, similarity in raw_results:
            rec = record_by_id.get(face_id)
            if rec is None:
                continue
            output.append({
                "face_db_id":  face_id,
                "image_id":    rec.get("image_id", ""),
                "image_path":  rec.get("image_path", ""),
                "similarity":  round(float(similarity), 4),
                "age":         rec.get("age"),
                "gender":      rec.get("gender"),
                "emotion":     rec.get("emotion"),
                "attributes":  rec.get("attributes", {}),
                "bbox":        [rec.get("bbox_x1"), rec.get("bbox_y1"),
                                rec.get("bbox_x2"), rec.get("bbox_y2")],
            })

        logger.info(f"Hybrid search returned {len(output)} results.")
        return output

    def search_by_attributes_only(
        self,
        gender:     Optional[str]  = None,
        emotion:    Optional[str]  = None,
        age_min:    Optional[int]  = None,
        age_max:    Optional[int]  = None,
        attributes: Optional[Dict[str, bool]] = None,
        limit:      int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Pure attribute-based search (no embedding query).
        Returns matching face records directly from SQLite.
        """
        candidate_ids = self.attribute_filter.filter(
            gender=gender, emotion=emotion,
            age_min=age_min, age_max=age_max,
            attributes=attributes, limit=limit,
        )
        return self.attribute_filter.get_by_ids(candidate_ids[:limit])
