import torch
from PIL import Image
from app.models.rcnn_face_detector import RCNNFaceDetector
from app.core.config import settings
import os

def test_detector():
    # Initialize detector
    detector = RCNNFaceDetector()
    
    # Load a query image (using one from the uploads if available, or any raw image)
    # I'll just use a sample image if it exists or download one for testing
    test_img_path = r"c:\Users\kadig\OneDrive\Desktop\PROJECT\sample_test.jpg"
    
    # Download a sample face for testing if not exists
    if not os.path.exists(test_img_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/test_images/itziar.jpg"
        urllib.request.urlretrieve(url, test_img_path)
    
    img = Image.open(test_img_path).convert("RGB")
    
    # Run detection
    results = detector.detect(img)
    
    print(f"Detected {results['count']} faces.")
    
    if results['count'] > 0:
        face_img = results['face_images'][0]
        output_path = "fixed_face_crop.jpg"
        face_img.save(output_path)
        print(f"Saved fixed crop to {output_path}")
        
        # Also check the loose crop
        loose_img = results['face_images_loose'][0]
        loose_path = "fixed_face_loose.jpg"
        loose_img.save(loose_path)
        print(f"Saved fixed loose crop to {loose_path}")
    else:
        print("No faces detected in test image.")

if __name__ == "__main__":
    test_detector()
