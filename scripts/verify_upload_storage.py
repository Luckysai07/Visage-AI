import os
import sys
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pipeline.face_pipeline import FacePipeline
from app.core.config import settings

def verify_direct():
    print("🚀 Starting Direct Pipeline Verification...")
    
    # 1. Initialize Pipeline
    pipeline = FacePipeline()
    
    # 2. Get initial count
    count_before = pipeline.attr_filter.count()
    print(f"📊 Faces in DB before: {count_before}")

    # 3. Use generated face image
    test_image = r"C:\Users\kadig\.gemini\antigravity\brain\5f62590a-ae9b-4cd9-aed0-ce39ae0f9c95\test_face_aligned_1773427302203.png"
    if not os.path.exists(test_image):
        print(f"❌ Test image {test_image} not found.")
        return
    
    pil_img = Image.open(test_image).convert("RGB")

    # 4. Run analyze with store_in_db=True
    print(f"🧪 Running pipeline.analyze with store_in_db=True...")
    result = pipeline.analyze(pil_img, store_in_db=True)
    
    print(f"✅ Analysis complete. Face count: {result['face_count']}")

    # 5. Check count after
    count_after = pipeline.attr_filter.count()
    print(f"📊 Faces in DB after: {count_after}")

    if count_after > count_before:
        print("✨ SUCCESS: Database count increased!")
    else:
        print("❌ FAILURE: Database count did not increase.")
        # Print first face result for debugging
        if result['faces']:
            print(f"Debug first face: {result['faces'][0].get('db_id')}")

if __name__ == "__main__":
    verify_direct()
