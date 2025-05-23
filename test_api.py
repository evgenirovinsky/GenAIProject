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
    sentences = [
        "Я дуже радію зустрічі з тобою.",
        "Сьогодні дуже сумно, бо йде дощ.",
        "Я здивований, що ти прийшов.",
        "Це просто нейтральне повідомлення.",
        "Я дуже злий, бо не встиг на автобус.",
        "Я дуже боюся, що не встигну здати роботу.",
        "Я дуже вдячний тобі за твою допомогу.",
        "Я дуже гордий, що ти досяг успіху.",
        "Я дуже задоволений, що ти прийшов.",
        "Я дуже засмучений, що ти не прийшов."
    ]
    for (i, sentence) in enumerate(sentences, 1):
        print(f"\nSentence {i}: “{sentence}”")
        data = { "text": sentence, "use_ukrainian": True, "threshold": 0.2 }
        try:
            response = requests.post(url, headers=headers, json=data)
            print("Response status code:", response.status_code)
            print("Response body:", json.dumps(response.json(), indent=2, ensure_ascii=False))
        except Exception as e:
            print("Error:", e)

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