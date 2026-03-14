import os
import sys
import logging
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def diagnose():
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    use_supabase = os.getenv("USE_SUPABASE", "False").lower() == "true"

    print("\n" + "="*50)
    print("      SUPABASE DIAGNOSTIC TOOL")
    print("="*50)
    
    if not use_supabase:
        print("❌ USE_SUPABASE is set to False in .env")
        print("   Change it to True if you want to use the cloud database.")
        return

    print(f"🔗 URL: {url}")
    print(f"🔑 Key: {key[:15]}...{key[-5:] if key else ''}")

    if not url or not key:
        print("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env")
        return

    if key.startswith("sb_publishable_"):
        print("⚠️  WARNING: Your key starts with 'sb_publishable_'.")
        print("   This looks like a STRIPE key, not a Supabase key!")
        print("   Please double check your .env file.")

    try:
        client: Client = create_client(url, key)
        print("✅ Supabase client initialized.")
        
        # Test 1: Check if table 'faces' exists
        print("\nChecking 'faces' table...")
        try:
            res = client.table("faces").select("id").limit(1).execute()
            print("✅ 'faces' table exists.")
            
            # Test 2: Check for storage_url column
            print("\nChecking schema for 'storage_url'...")
            try:
                # We try to select it
                res = client.table("faces").select("storage_url").limit(1).execute()
                print("✅ 'storage_url' column found.")
            except Exception:
                print("❌ 'storage_url' column is MISSING.")
                print("   Run this SQL in Supabase Dashboard (SQL Editor):")
                print("   ALTER TABLE faces ADD COLUMN storage_url TEXT;")

        except Exception as e:
            if "relation \"public.faces\" does not exist" in str(e):
                print("❌ Table 'faces' DOES NOT EXIST.")
                print("   Please create it in the Supabase Dashboard SQL Editor.")
                print("   (See full SQL script below)")
            else:
                print(f"❌ Error accessing table: {e}")

        # Test 3: Check Storage Bucket
        bucket_name = "face-images"
        print(f"\nChecking Storage Bucket '{bucket_name}'...")
        try:
            # Note: anon keys often can't 'list_buckets', but CAN access them if policies are set
            # We'll try to get the bucket directly
            bucket = client.storage.get_bucket(bucket_name)
            if bucket:
                print(f"✅ Bucket '{bucket_name}' found.")
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"❌ Bucket '{bucket_name}' NOT FOUND.")
                print(f"   Go to Supabase Storage and create a public bucket named '{bucket_name}'.")
            else:
                print(f"⚠️  Note: Could not list/get bucket details with this key: {e}")
                print("   This is common for 'anon' keys. If you created the bucket, it's likely fine.")
                print("   The app will try to upload anyway.")

    except Exception as e:
        print(f"❌ Connection failed: {e}")

    print("\n" + "="*50)
    print("      🚀 FINAL FIX: RUN THIS IN SUPABASE SQL EDITOR")
    print("="*50)
    print("""
-- 1. FIX DATABASE PERMISSIONS
-- This should work because you created the 'faces' table
ALTER TABLE faces DISABLE ROW LEVEL SECURITY;

-- 2. FIX STORAGE PERMISSIONS (Smarter way)
-- We use a POLICY instead of ALTER TABLE to avoid permission errors
CREATE POLICY "Allow Public Uploads" 
ON storage.objects FOR ALL 
USING ( bucket_id = 'face-images' )
WITH CHECK ( bucket_id = 'face-images' );

-- 3. ENSURE BUCKET IS PUBLIC
-- Go to Storage -> face-images -> Edit Bucket -> Toggle "Public" to ON.
""")
    print("="*50)

if __name__ == "__main__":
    diagnose()
