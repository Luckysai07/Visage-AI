import requests
import json

API_URL = "http://localhost:8001/api/search"

def test_search():
    try:
        # Try attribute-only search first
        params = {
            "gender": "male",
            "top_k": 5
        }
        print(f"Testing attribute-only search: {params}")
        response = requests.post(API_URL, data=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"Found {len(results)} results.")
            for i, res in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  image_id: {res.get('image_id')}")
                print(f"  storage_url: {res.get('storage_url')}")
                print(f"  gender: {res.get('gender')}")
        else:
            print(f"Search failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    test_search()
