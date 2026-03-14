import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.retrieval.faiss_index import FaissIndex
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.attribute_filter import AttributeFilter
from app.core.config import settings

def test_synthetic_search():
    print("🚀 Starting Synthetic Search Benchmark...")
    
    # Initialize components
    db = AttributeFilter()
    index = FaissIndex()
    searcher = HybridSearch(index, db)
    
    # Check if index has data
    if index.size == 0:
        print("ℹ️ Index empty, adding synthetic data for benchmark...")
        num_vecs = 100
        dim = settings.EMBEDDING_DIM
        
        # Generate random normalized embeddings
        vecs = np.random.randn(num_vecs, dim).astype('float32')
        vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]
        
        ids = list(range(1, num_vecs + 1))
        
        # We need to simulate DB entries too
        for i in ids:
            db.insert_face(
                image_path=f"synthetic_{i}.jpg",
                image_id=f"syn_{i}",
                age=25, gender="male", gender_confidence=0.9,
                emotion="happy", emotion_confidence=0.9,
                attributes={"Smiling": True},
                bbox=(0,0,100,100), detection_confidence=0.9,
                faiss_position=i-1
            )
        
        index.add(vecs, ids)
        print(f"Added {num_vecs} synthetic faces.")

    # 1. Test Confidence Scaling Logic
    print("\n--- Testing Confidence Calibration ---")
    test_sims = [0.40, 0.50, 0.65, 0.70, 0.85, 0.95]
    for sim in test_sims:
        conf = searcher._calculate_match_confidence(sim)
        print(f"  Similarity: {sim:.2f} -> Match Confidence: {conf:.2f}%")

    # 2. Benchmark Search Speed
    print("\n--- Benchmarking Search Performance ---")
    query = np.random.randn(settings.EMBEDDING_DIM).astype('float32')
    query /= np.linalg.norm(query)
    
    start_time = time.time()
    results = searcher.search(query, k=5)
    latency = (time.time() - start_time) * 1000
    
    print(f"Search Results: {len(results)}")
    print(f"Total Search Latency: {latency:.2f} ms")
    
    if results:
        print(f"Top Match Confidence: {results[0]['match_confidence']}%")

    print("\n✅ Synthetic Verification Complete.")

if __name__ == "__main__":
    test_synthetic_search()
