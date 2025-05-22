# Ukrainian Text Emotion Analysis System

A powerful system for analyzing emotions and sentiment in Ukrainian text using state-of-the-art transformer models. This system combines sentiment analysis with detailed emotion detection to provide comprehensive analysis of Ukrainian text.

## Features

- **Sentiment Analysis**: Classifies text into positive, negative, or neutral sentiment using a pre-trained Ukrainian BERT model
- **Emotion Detection**: Identifies specific emotions (joy, sadness, anger, fear, surprise, neutral) using a multilingual RoBERTa model
- **Confidence Scores**: Provides confidence scores for both sentiment and emotion predictions
- **Batch Processing**: Supports analyzing multiple texts efficiently
- **REST API**: Includes a FastAPI-based API for easy integration
- **Ukrainian Language Support**: Optimized for Ukrainian text with Ukrainian emotion labels

## Prerequisites

1. Python 3.8 or higher
2. pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ukrainian-emotion-analysis
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Direct Usage

You can use the system directly in your Python code:

```python
from emotion_analyzer.analyzer import EmotionAnalysisSystem

analyzer = EmotionAnalysisSystem()

# Analyze a single text
text = "Я дуже радий зустріти вас! Це чудовий день!"
result = analyzer.analyze(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Dominant Emotion: {result['emotions']['dominant_emotion']}")
print("Emotion Mixture:", ", ".join(result['emotions']['emotion_mixture']))

# Get confidence scores
scores = analyzer.get_confidence_scores(text)
print("\nConfidence Scores:")
print("Sentiment scores:", scores['sentiment'])
print("Emotion scores:", scores['emotions'])

# Analyze multiple texts
texts = [
    "Я дуже радий зустріти вас!",
    "Мені дуже сумно, що ви не прийшли.",
    "Я здивований вашою реакцією."
]
results = analyzer.analyze_batch(texts)
```

### Using the API

1. Start the API server:
```bash
python -m api.main
```

2. The API will be available at `http://localhost:8000` with the following endpoints:

- `POST /analyze`: Analyze a single text
  ```bash
  curl -X POST "http://localhost:8000/analyze" \
       -H "Content-Type: application/json" \
       -d '{"text": "Я дуже радий зустріти вас!"}'
  ```

- `POST /analyze/batch`: Analyze multiple texts
  ```bash
  curl -X POST "http://localhost:8000/analyze/batch" \
       -H "Content-Type: application/json" \
       -d '{"texts": ["Текст 1", "Текст 2"]}'
  ```

- `POST /analyze/sentiment`: Analyze only sentiment
- `POST /analyze/emotions`: Analyze only emotions
- `GET /health`: Health check endpoint

See the API documentation at `http://localhost:8000/docs` for detailed endpoint information.

## Project Structure

```
ukrainian-emotion-analysis/
├── emotion_analyzer/
│   ├── __init__.py
│   ├── analyzer.py      # Main analyzer class
│   ├── sentiment.py     # Sentiment analysis module
│   └── emotions.py      # Emotion detection module
├── api/
│   ├── __init__.py
│   └── main.py         # FastAPI application
├── examples/
│   ├── __init__.py
│   ├── basic_usage.py  # Direct usage examples
│   └── api_usage.py    # API usage examples
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Models Used

1. **Sentiment Analysis**: `cointegrated/rubert-tiny2-cedr-emotion-detection`
   - A publicly available Russian/Ukrainian BERT model fine-tuned for emotion detection
   - The model outputs the following emotion labels: `anger`, `fear`, `joy`, `no_emotion`, `sadness`, `surprise`
   - **Sentiment mapping:**
     - `joy` → positive
     - `sadness`, `anger`, `fear` → negative
     - `no_emotion`, `surprise` → neutral
   - **No Hugging Face token is required** for this model

2. **Emotion Detection**: `j-hartmann/emotion-english-distilroberta-base`
   - A multilingual RoBERTa model fine-tuned for emotion analysis
   - Detects six emotions: joy, sadness, anger, fear, surprise, and neutral
   - Provides Ukrainian translations for emotion labels

## Troubleshooting

### Model Access Issues

- The sentiment model is now public and does **not** require authentication or a Hugging Face token.
- If you encounter errors related to model loading, ensure you have a stable internet connection and the latest version of `transformers` and `torch` installed.
- If you see errors about missing labels or keys, make sure your code is up to date and matches the label mapping described above.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The sentiment analysis model is based on the work of cointegrated
- The emotion detection model is based on the work of j-hartmann
- Thanks to the Hugging Face team for the Transformers library 