import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.face_detector import FaceDetector
from app.models.age_gender_model import AgeGenderModel
from app.core.config import settings

def run_diagnostic(image_path):
    detector = FaceDetector()
    ag_model = AgeGenderModel()
    
    img = Image.open(image_path).convert("RGB")
    detection = detector.detect(img)
    
    if detection["count"] == 0:
        print("No faces detected.")
        return

    results_dir = Path("data/results/diagnostic")
    results_dir.mkdir(parents=True, exist_ok=True)

    for i in range(detection["count"]):
        face_aligned = detection["face_images"][i]
        face_loose = detection["face_images_loose"][i]
        
        # Predict on both for comparison
        res_aligned = ag_model.predict(face_aligned)
        res_loose = ag_model.predict(face_loose)
        
        print(f"Face {i}:")
        print(f"  Aligned Crop -> Age: {res_aligned['age']}, Gender: {res_aligned['gender']}")
        print(f"  Loose Crop   -> Age: {res_loose['age']}, Gender: {res_loose['gender']}")
        
        # Save crops for manual inspection
        face_aligned.save(results_dir / f"face_{i}_aligned.jpg")
        face_loose.save(results_dir / f"face_{i}_loose.jpg")
        
    print(f"\nDiagnostic images saved to {results_dir}")

if __name__ == "__main__":
    # Use a sample image from the project if available, or ask user
    test_img = "data/raw/sample.jpg" # Update this path if needed
    if not os.path.exists(test_img):
        print(f"Please provide a test image at {test_img} or modify the script.")
    else:
        run_diagnostic(test_img)
