import requests
import json
import time

def test_api():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "text": "Where is my order?",
        "model": "tfidf_lr",
        "top_k": 3
    }
    
    print(f"Sending request to {url}...")
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=30)
        duration = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        print(f"Duration: {duration:.2f}s")
        
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error:", response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
