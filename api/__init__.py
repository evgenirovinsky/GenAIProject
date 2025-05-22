"""
Ukrainian Emotion Analysis API

This package provides a FastAPI-based REST API for the Ukrainian Text Emotion Analysis System.
It exposes endpoints for analyzing emotions and sentiment in Ukrainian text.

Available endpoints:
- /analyze: Analyze a single text for both sentiment and emotions
- /analyze/batch: Analyze multiple texts in batch
- /analyze/sentiment: Analyze only the sentiment of a text
- /analyze/emotions: Analyze only the emotions in a text
- /health: Health check endpoint

Example usage:
    import requests
    
    response = requests.post(
        "http://localhost:8000/analyze",
        json={"text": "Я дуже радий зустріти вас!"}
    )
    print(response.json())
"""

__version__ = "1.0.0" 