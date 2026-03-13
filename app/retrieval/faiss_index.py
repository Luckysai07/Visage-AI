"""
Phase 3 — FAISS Vector Index Wrapper.
Manages the face embedding index for similarity search.
Uses IndexFlatIP (inner product) on L2-normalized vectors
which is equivalent to cosine similarity.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss

from app.core.config import settings

logger = logging.getLogger(__name__)


class FaissIndex:
    """
    FAISS-based approximate nearest neighbor search index.

    Design:
    - IndexFlatIP: exact search via inner product (= cosine on L2-normalized vecs)
    - An external id_map (np.ndarray) maps FAISS row positions → SQLite image_ids
    - index.add() appends vectors; positions are 0-indexed
    """

    def __init__(self, dim: int = None):
        self.dim = dim or settings.EMBEDDING_DIM
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: List[int] = []   # position i → SQLite image_id
        self._initialize()

    def _initialize(self):
        """Create a fresh inner-product index."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []
        logger.debug(f"Initialized FAISS IndexFlatIP (dim={self.dim})")

    # ── Adding vectors ─────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, image_ids: List[int]):
        """
        Add L2-normalized embeddings to the index.

        Args:
            embeddings: float32 array of shape (N, dim).
            image_ids:  Corresponding SQLite row IDs (length N).
        """
        assert embeddings.shape[1] == self.dim, \
            f"Expected dim={self.dim}, got {embeddings.shape[1]}"
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.id_map.extend(image_ids)
        logger.debug(f"Added {len(image_ids)} vectors. Total: {self.index.ntotal}")

    def add_single(self, embedding: np.ndarray, image_id: int):
        """Add a single embedding vector."""
        self.add(embedding.reshape(1, -1), [image_id])

    # ── Searching ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        candidate_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find the top-K most similar embeddings.

        Args:
            query:         float32 array of shape (dim,) — L2-normalized query.
            k:             Number of results to return.
            candidate_ids: If provided, restrict search to these SQLite image_ids.
                           Used for hybrid (attribute-filtered) search.

        Returns:
            List of (image_id, similarity_score) sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        query_vec = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)

        if candidate_ids is not None:
            return self._restricted_search(query_vec, k, candidate_ids)

        actual_k = min(k, self.index.ntotal)
        scores, positions = self.index.search(query_vec, actual_k)

        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:   # FAISS padding value
                continue
            image_id = self.id_map[pos]
            results.append((image_id, float(score)))
        return results

    def _restricted_search(
        self,
        query_vec: np.ndarray,
        k: int,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Search only among a subset of indexed vectors (for attribute filtering).
        Extracts candidate vectors, searches locally.
        """
        candidate_set = set(candidate_ids)
        # Map candidate SQLite IDs → FAISS positions
        candidate_positions = [
            pos for pos, img_id in enumerate(self.id_map)
            if img_id in candidate_set
        ]
        if not candidate_positions:
            return []

        # Extract candidate embeddings
        all_vecs = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.dim)
        all_vecs = np.array(all_vecs, dtype=np.float32).reshape(self.index.ntotal, self.dim)
        candidate_vecs = all_vecs[candidate_positions]   # (M, dim)

        # Brute-force inner product among candidates
        similarities = (query_vec @ candidate_vecs.T)[0]   # (M,)
        actual_k = min(k, len(candidate_positions))
        top_local = np.argsort(-similarities)[:actual_k]

        results = []
        for local_idx in top_local:
            faiss_pos  = candidate_positions[local_idx]
            image_id   = self.id_map[faiss_pos]
            score      = float(similarities[local_idx])
            results.append((image_id, score))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path = None,
        id_map_path: Path = None,
    ):
        """Save FAISS index and ID map to disk."""
        index_path  = index_path  or settings.FAISS_INDEX_PATH
        id_map_path = id_map_path or settings.FAISS_ID_MAP_PATH
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.save(str(id_map_path), np.array(self.id_map, dtype=np.int64))
        logger.info(f"FAISS index saved: {index_path} ({self.index.ntotal} vectors)")

    def load(
        self,
        index_path: Path = None,
        id_map_path: Path = None,
    ) -> bool:
        """Load FAISS index and ID map from disk. Returns True on success."""
        index_path  = index_path  or settings.FAISS_INDEX_PATH
        id_map_path = id_map_path or settings.FAISS_ID_MAP_PATH
        if not index_path.exists() or not id_map_path.exists():
            logger.warning("FAISS index files not found. Run build-database first.")
            return False
        self.index  = faiss.read_index(str(index_path))
        self.id_map = np.load(str(id_map_path)).tolist()
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        return True

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        return self.index.ntotal if self.index else 0

    def reset(self):
        """Clear the index."""
        self._initialize()
        logger.info("FAISS index reset.")
