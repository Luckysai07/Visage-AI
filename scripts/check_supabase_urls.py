import os
from supabase import create_client, Client
from dotenv import load_dotenv

def check_supabase():
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("Supabase credentials not found in .env")
        return

    try:
        supabase: Client = create_client(url, key)
        res = supabase.table("faces").select("id, image_id, storage_url").limit(5).execute()
        
        with open("supabase_urls.txt", "w") as f:
            f.write(f"Retrieved {len(res.data)} records from Supabase:\n")
            for row in res.data:
                f.write("-" * 20 + "\n")
                f.write(f"ID: {row['id']}\n")
                f.write(f"ImageID: {row['image_id']}\n")
                f.write(f"URL: {row['storage_url']}\n")
        print("Results written to supabase_urls.txt")
            
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")

if __name__ == "__main__":
    check_supabase()
