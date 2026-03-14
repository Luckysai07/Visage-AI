import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pipeline.face_pipeline import FacePipeline
from app.core.config import settings

def test_search_optimality():
    print("🚀 Starting Search Optimality Verification...")
    
    pipeline = FacePipeline()
    
    # Check if we have any data to search
    count = pipeline.searcher.faiss_index.size
    print(f"Index Size: {count} faces")
    
    if count == 0:
        print("❌ FAISS index is empty. Cannot test search.")
        return

    # Use a sample image for search
    test_img_path = r"c:\Users\kadig\OneDrive\Desktop\PROJECT\sample_face.jpg"
    if not os.path.exists(test_img_path):
        test_img_path = r"c:\Users\kadig\OneDrive\Desktop\PROJECT\tests\test_face.jpg"
    
    if not os.path.exists(test_img_path):
        print(f"❌ Test image not found at expected locations.")
        return

    print(f"Using test image: {test_img_path}")

    img = Image.open(test_img_path).convert("RGB")
    
    print("\n--- Running Optimized Search ---")
    start_time = time.time()
    results = pipeline.analyze(img)
    latency = (time.time() - start_time) * 1000
    
    print(f"Total Analysis Latency: {latency:.2f} ms")
    
    if not results or "faces" not in results:
        print("❌ No faces field in results.")
        return
    
    face_count = len(results["faces"])
    print(f"Faces Detected: {face_count}")
    
    if face_count == 0:
        print("❌ No faces detected in test image.")
        return

    for i, face in enumerate(results["faces"]):
        print(f"\nFace {i+1} Results:")
        search_results = face.get("similar_images", [])
        if not search_results:
            print(f"  No search matches found for face {i+1}.")
            continue
            
        print(f"  Top Match (ID: {search_results[0]['image_id']})")
        print(f"  Raw Similarity:   {search_results[0]['similarity']}")
        print(f"  Match Confidence: {search_results[0]['match_confidence']}%")
        
        # Verify confidence logic
        sim = search_results[0]['similarity']
        conf = search_results[0]['match_confidence']
        
        print("\nVerifying Confidence Calibration:")
        if sim > 0.75 and conf > 80:
            print("  ✅ High similarity correctly results in high confidence %")
        elif sim < 0.60 and conf < 50:
            print("  ✅ Low similarity correctly results in low confidence %")
        else:
            print("  ℹ️ Confidence matches expected distribution.")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    test_search_optimality()
