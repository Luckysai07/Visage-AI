import os
import sys
import uuid
import numpy as np
from PIL import Image
import io
from supabase import create_client, Client
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.retrieval.supabase_filter import SupabaseFilter

def test_insert():
    load_dotenv()
    print("Starting Test Insert...")
    
    try:
        sb = SupabaseFilter()
        
        # 1. Test Data
        test_id = str(uuid.uuid4())
        
        # 2. Test Image
        test_img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        
        # 3. Try Upload
        print(f"Testing Upload for image {test_id}...")
        url = sb.upload_face_image(test_img, f"test_{test_id}")
        if url:
            print(f"✅ Upload success: {url}")
        else:
            print("❌ Upload failed (check bucket existence and permissions)")

        # 4. Try Database Insert
        print(f"Testing DB Insert for {test_id}...")
        res = sb.insert_face(
            image_path="test_path.jpg",
            image_id=test_id,
            age=25,
            gender="Male",
            gender_confidence=0.99,
            emotion="Happy",
            emotion_confidence=0.9,
            attributes={"Smiling": True},
            bbox=(10, 10, 90, 90),
            detection_confidence=0.95,
            faiss_position=9999,
            storage_url=url
        )
        
        if res != -1:
            print(f"✅ DB Insert success! Row ID: {res}")
            # Clean up
            # sb.client.table("faces").delete().eq("image_id", test_id).execute()
            # print("✅ Cleaned up test record.")
        else:
            print("❌ DB Insert failed.")

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_insert()
