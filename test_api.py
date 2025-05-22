import requests
import json
import threading
import time
import uvicorn
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_server():
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)

def test_analyze_endpoint():
    url = "http://localhost:8000/analyze"
    headers = {"Content-Type": "application/json"}
    data = {
        "text": "Я дуже радий зустріти вас!",
        "use_ukrainian": True,
        "threshold": 0.2
    }
    
    # Wait for server to start
    max_retries = 5
    retry_delay = 2
    
    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print("Response status code:", response.status_code)
            print("Response body:", json.dumps(response.json(), indent=2, ensure_ascii=False))
            return
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                print(f"Attempt {i + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Error:", e)
                if hasattr(e, 'response') and e.response is not None:
                    print("Error details:", e.response.text)

if __name__ == "__main__":
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start
    print("Starting server...")
    time.sleep(5)
    
    # Run the test
    print("Testing API endpoint...")
    test_analyze_endpoint()
    
    # Keep the main thread alive for a moment to see the results
    time.sleep(2) 