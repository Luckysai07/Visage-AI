import os
import sys
import torch
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.face_detector import FaceDetector
from app.models.age_gender_model import AgeGenderModel
from app.core.config import settings

def verify_optimization():
    print("🚀 Starting Age Optimization Verification (Synthetic)...")
    
    ag_model = AgeGenderModel()
    
    # Create two synthetic face images (different colors/patterns)
    img1 = Image.new('RGB', (224, 224), color=(100, 100, 100)) # Grey
    img2 = Image.new('RGB', (224, 224), color=(200, 200, 200)) # Light Grey
    
    # 1. Test Single Crop 1
    print("\n--- Single Crop 1 ---")
    res1 = ag_model.predict(img1)
    print(f"Age 1: {res1['age']} (raw: {res1['age_raw']})")
    
    # 2. Test Single Crop 2
    print("\n--- Single Crop 2 ---")
    res2 = ag_model.predict(img2)
    print(f"Age 2: {res2['age']} (raw: {res2['age_raw']})")
    
    # 3. Test Dual-Crop (Blended)
    print("\n--- Dual-Crop (Blended) ---")
    res_dual = ag_model.predict(img1, aligned_face=img2)
    print(f"Age Dual: {res_dual['age']} (raw: {res_dual['age_raw']})")
    
    # Verification of blending
    expected_raw = round((res1['age_raw'] + res2['age_raw']) / 2.0, 2)
    print(f"\nCalculated Expected Raw: {expected_raw}")
    
    # Use a small epsilon for floating point comparison
    if abs(res_dual['age_raw'] - expected_raw) < 0.1:
        print("✅ SUCCESS: Dual-Crop prediction is correctly averaging the crops!")
    else:
        print(f"❌ FAILURE: Dual-Crop prediction ({res_dual['age_raw']}) mismatch with expected ({expected_raw})")

if __name__ == "__main__":
    verify_optimization()

if __name__ == "__main__":
    verify_optimization()
