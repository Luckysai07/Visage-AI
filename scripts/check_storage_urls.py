import sqlite3
from pathlib import Path
import os
import json

def check_db():
    db_path = Path(r"c:\Users\kadig\OneDrive\Desktop\PROJECT\data\database\face_database.db")
    if not db_path.exists():
        print(f"DB not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT id, image_id, storage_url FROM faces LIMIT 10")
    rows = cursor.fetchall()
    
    print(f"Checking first {len(rows)} records:")
    for row in rows:
        print(f"ID: {row['id']}, ImageID: {row['image_id']}, StorageURL: {row['storage_url']}")
    
    conn.close()

if __name__ == "__main__":
    check_db()
