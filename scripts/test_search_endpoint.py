import requests
import os

API_URL = "http://localhost:8001/api/search"

def test_attr_search():
    print("🔍 Testing Attribute-Only Search...")
    data = {
        "gender": "male",
        "top_k": 5
    }
    try:
        response = requests.post(API_URL, data=data)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Success! Found {results.get('count')} results.")
            for i, res in enumerate(results.get('results', [])):
                print(f"  {i+1}. Result: {res}")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {e}")

def test_hybrid_search():
    print("\n🔍 Testing Hybrid Search (with image)...")
    test_image = "tests/test_face.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image {test_image} not found.")
        return

    with open(test_image, "rb") as f:
        files = {"file": f}
        data = {"top_k": 5}
        try:
            response = requests.post(API_URL, files=files, data=data)
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Success! Found {results.get('count')} results.")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_attr_search()
    test_hybrid_search()
