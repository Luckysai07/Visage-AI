"""
Phase 2 Level 3 — Deep Face Embedding Extractor.
Uses facenet-pytorch InceptionResnetV1 pretrained on VGGFace2.
Produces 512-dim L2-normalized embedding vectors for similarity search.
"""

import logging
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from app.core.config import settings
from app.core.device import DEVICE
from app.utils.image_utils import face_preprocess_transform

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    ArcFace-style face embedding extractor using InceptionResnetV1
    pretrained on VGGFace2 (512-dimensional embeddings).

    Embeddings are L2-normalized so that cosine similarity equals
    dot product, enabling FAISS IndexFlatIP for efficient retrieval.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        # Pretrained on VGGFace2 — downloads weights automatically (~100MB)
        self.model = InceptionResnetV1(pretrained="vggface2").to(self.device)
        self.model.eval()
        logger.info(f"EmbeddingModel (InceptionResnetV1 / VGGFace2) ready on {self.device}")

    @torch.no_grad()
    def extract(self, face_image: Image.Image) -> np.ndarray:
        """
        Extract a 512-dim L2-normalized embedding from a face image.

        Args:
            face_image: PIL RGB face image (will be resized to 160x160 internally).

        Returns:
            np.ndarray of shape (512,), dtype float32, L2-normalized.
        """
        tensor = face_preprocess_transform(face_image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)                    # (1, 512)
        embedding = F.normalize(embedding, p=2, dim=1)   # L2-normalize
        return embedding[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, face_images: List[Image.Image]) -> np.ndarray:
        """
        Extract embeddings for a batch of face images.

        Args:
            face_images: List of PIL RGB face images.

        Returns:
            np.ndarray of shape (N, 512), float32, each row L2-normalized.
        """
        if not face_images:
            return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

        tensors = torch.stack([
            face_preprocess_transform(img) for img in face_images
        ]).to(self.device)

        embeddings = self.model(tensors)                  # (N, 512)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_from_tensor(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract embedding from a preprocessed MTCNN face tensor (3, 160, 160)
        already normalized to [-1, 1].

        Returns:
            np.ndarray of shape (512,), float32, L2-normalized.
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        embedding = self.model(face_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0].cpu().numpy().astype(np.float32)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        Since embeddings are L2-normalized, cosine_sim = dot product.

        Returns:
            float in [-1, 1], where 1.0 = identical face.
        """
        return float(np.dot(emb1, emb2))
