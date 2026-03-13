"""
Supabase-backed metadata store and attribute filter.
Provides cloud persistence and PostgreSQL JSONB-based filtering.
"""

import logging
import io
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from supabase import create_client, Client

from app.core.config import settings

logger = logging.getLogger(__name__)

class SupabaseFilter:
    """
    Supabase (PostgreSQL) implementation of the attribute filter.
    Allows for cloud scaling of metadata.
    """

    def __init__(self, url: str = None, key: str = None):
        self.url = url or settings.SUPABASE_URL
        self.key = key or settings.SUPABASE_KEY
        
        if not self.url or not self.key:
            logger.warning("Supabase URL or Key not provided. Filter will likely fail.")
            self.client = None
        else:
            self.client: Client = create_client(self.url, self.key)
            logger.info("Supabase client initialized.")

    def insert_face(
        self,
        image_path: str,
        image_id: str,
        age: Optional[int],
        gender: Optional[str],
        gender_confidence: Optional[float],
        emotion: Optional[str],
        emotion_confidence: Optional[float],
        attributes: Optional[Dict[str, Any]],
        bbox: Optional[Tuple[float, float, float, float]],
        detection_confidence: Optional[float],
        faiss_position: int,
        storage_url: Optional[str] = None,
    ) -> int:
        """Insert a face record into Supabase."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized.")

        bbox_vals = bbox if bbox else (None, None, None, None)
        
        data = {
            "image_path": image_path,
            "image_id": image_id,
            "age": age,
            "gender": gender,
            "gender_confidence": gender_confidence,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "attributes": attributes, # Supabase client handles dict to JSONB
            "bbox_x1": bbox_vals[0],
            "bbox_y1": bbox_vals[1],
            "bbox_x2": bbox_vals[2],
            "bbox_y2": bbox_vals[3],
            "detection_confidence": detection_confidence,
            "faiss_position": faiss_position,
            "storage_url": storage_url,
        }

        try:
            # upsert based on image_id
            res = self.client.table("faces").upsert(data, on_conflict="image_id").execute()
            
            if len(res.data) > 0:
                logger.info(f"Successfully stored face {image_id} in Supabase.")
                return res.data[0]["id"]
            return -1
        except Exception as e:
            logger.error(f"Supabase Insertion Error for face {image_id}: {e}")
            if "PGRST301" in str(e):
                logger.error("TIP: Your Supabase Key might be invalid or expired.")
            elif "404" in str(e):
                logger.error("TIP: The 'faces' table was not found. Please create it in the SQL Editor.")
            return -1

    def filter(
        self,
        gender:         Optional[str]  = None,
        emotion:        Optional[str]  = None,
        age_min:        Optional[int]  = None,
        age_max:        Optional[int]  = None,
        attributes:     Optional[Dict[str, bool]] = None,
        limit:          int = 1000,
    ) -> List[int]:
        """
        Filter database records by attribute constraints using Supabase.
        """
        if not self.client:
            return []

        query = self.client.table("faces").select("id, attributes")

        if gender:
            query = query.ilike("gender", gender)
        if emotion:
            query = query.ilike("emotion", emotion)
        if age_min is not None:
            query = query.gte("age", age_min)
        if age_max is not None:
            query = query.lte("age", age_max)
            
        # Limit at DB level
        query = query.limit(limit)
        
        res = query.execute()
        rows = res.data

        if not attributes:
            return [row["id"] for row in rows]

        # Post-filter for attributes (JSONB matching)
        # Note: We could do this using postgres arrow operators if needed, 
        # but for simplicity and feature parity with local, we do it here.
        filtered_ids: List[int] = []
        for row in rows:
            stored_attrs = row.get("attributes")
            if not stored_attrs:
                continue
            
            if all(
                stored_attrs.get(attr_name) == required_val
                for attr_name, required_val in attributes.items()
            ):
                filtered_ids.append(row["id"])

        return filtered_ids

    def get_by_id(self, face_id: int) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        res = self.client.table("faces").select("*").eq("id", face_id).execute()
        return res.data[0] if res.data else None

    def get_by_ids(self, face_ids: List[int]) -> List[Dict[str, Any]]:
        if not self.client or not face_ids:
            return []
        res = self.client.table("faces").select("*").in_("id", face_ids).execute()
        return res.data

    def count(self) -> int:
        if not self.client:
            return 0
        res = self.client.table("faces").select("id", count="exact").execute()
        return res.count if res.count is not None else 0

    def upload_face_image(self, face_image: Image.Image, image_id: str) -> Optional[str]:
        """
        Upload a face crop to Supabase Storage and return the public URL.
        """
        if not self.client:
            return None

        try:
            # Convert PIL to bytes
            img_byte_arr = io.BytesIO()
            face_image.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()

            path = f"{image_id}.jpg"
            bucket = settings.SUPABASE_BUCKET

            # Upload to Supabase Storage
            self.client.storage.from_(bucket).upload(
                path=path,
                file=img_bytes,
                file_options={"content-type": "image/jpeg"}
            )

            # Get public URL
            res = self.client.storage.from_(bucket).get_public_url(path)
            # In some versions of supabase-py, get_public_url returns a string directly.
            # In others, it returns an object with a public_url property.
            if hasattr(res, 'public_url'):
                return res.public_url
            return str(res)
        except Exception as e:
            logger.error(f"Failed to upload image to Supabase: {e}")
            return None

    def clear(self):
        if not self.client:
            return
        # Bulk delete
        self.client.table("faces").delete().neq("id", -1).execute()
        logger.info("Supabase 'faces' table cleared.")
