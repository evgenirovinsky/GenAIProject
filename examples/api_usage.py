import requests
import json
from typing import Dict, List, Any

API_URL = "http://localhost:8000"

def analyze_text(text: str, endpoint: str = "/analyze") -> Dict[str, Any]:
    """
    Send a text to the API for analysis.
    """
    response = requests.post(
        f"{API_URL}{endpoint}",
        json={"text": text, "use_ukrainian": True}
    )
    response.raise_for_status()
    return response.json()

def analyze_batch(texts: List[str]) -> Dict[str, Any]:
    """
    Send multiple texts to the API for batch analysis.
    """
    response = requests.post(
        f"{API_URL}/analyze/batch",
        json={"texts": texts, "use_ukrainian": True}
    )
    response.raise_for_status()
    return response.json()

def main():
    # Example texts in Ukrainian
    texts = [
        "Я дуже радий зустріти вас! Це чудовий день!",
        "Мені дуже сумно, що ви не прийшли на зустріч.",
        "Я здивований вашою реакцією на цю новину.",
        "Це просто нейтральне повідомлення без особливих емоцій."
    ]

    print("Testing API Endpoints")
    print("=" * 50)

    # Test health check
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print("\nHealth Check:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        return

    # Test individual text analysis
    print("\nAnalyzing Individual Texts:")
    print("-" * 50)
    
    for text in texts:
        print(f"\nText: {text}")
        
        # Full analysis
        try:
            result = analyze_text(text)
            print("\nFull Analysis:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except requests.exceptions.RequestException as e:
            print(f"Error in full analysis: {e}")
            continue

        # Sentiment analysis
        try:
            result = analyze_text(text, "/analyze/sentiment")
            print("\nSentiment Analysis:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except requests.exceptions.RequestException as e:
            print(f"Error in sentiment analysis: {e}")
            continue

        # Emotion analysis
        try:
            result = analyze_text(text, "/analyze/emotions")
            print("\nEmotion Analysis:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except requests.exceptions.RequestException as e:
            print(f"Error in emotion analysis: {e}")
            continue

        print("-" * 50)

    # Test batch processing
    print("\nBatch Processing:")
    print("-" * 50)
    
    try:
        results = analyze_batch(texts)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print(f"Error in batch processing: {e}")

if __name__ == "__main__":
    main() 