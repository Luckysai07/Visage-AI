"""
End-to-End Pipeline Tests.
Validates the full FacePipeline using a generated dummy image.
"""

import os
import pytest
import numpy as np
from PIL import Image

from app.core.config import settings
from app.pipeline.face_pipeline import FacePipeline

# Try to use a real face if available locally, else generate a blank dummy image
TEST_IMAGE_PATH = "tests/test_face.jpg"

@pytest.fixture(scope="module")
def pipeline():
    """Load the full pipeline once for all tests in this module."""
    # Ensure environment knows it's testing
    os.environ["TESTING"] = "1"
    
    # We create a dummy test_face.jpg so the suite can run even without raw data
    if not os.path.exists(TEST_IMAGE_PATH):
        os.makedirs("tests", exist_ok=True)
        # Create a blank RGB image. Note: MTCNN might return 0 faces for a blank image,
        # but the test will check the overall application structure.
        img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        img.save(TEST_IMAGE_PATH)
        
    return FacePipeline()

def test_pipeline_initialization(pipeline):
    """Test that all models load correctly."""
    assert pipeline.detector is not None
    assert pipeline.age_gender is not None
    assert pipeline.emotion is not None
    assert pipeline.attribute is not None
    assert pipeline.embedding is not None

def test_pipeline_analyze(pipeline):
    """Test the main analyze function."""
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    result = pipeline.analyze(image, top_k=2, generate_heatmap=False)
    
    assert "face_count" in result
    assert "detected_image" in result
    assert "faces" in result
    
    # If a real test face was provided and detected:
    if result["face_count"] > 0:
        face = result["faces"][0]
        assert "age" in face
        assert "gender" in face
        assert "emotion" in face
        assert "attributes" in face
        assert "similar_images" in face

def test_faiss_index_mock():
    """Test the FAISS wrapper directly."""
    from app.retrieval.faiss_index import FaissIndex
    index = FaissIndex(dim=settings.EMBEDDING_DIM)
    
    # Create two random normalized embeddings
    emb1 = np.random.randn(1, settings.EMBEDDING_DIM).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2 = np.random.randn(1, settings.EMBEDDING_DIM).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Add to index
    index.add(emb1, [100])
    assert index.size == 1
    
    # Search
    results = index.search(emb1[0], k=1)
    assert len(results) == 1
    assert results[0][0] == 100
    assert pytest.approx(results[0][1], 0.01) == 1.0  # Self similarity should be ~1
