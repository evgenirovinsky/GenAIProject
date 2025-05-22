from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_analyzer.analyzer import EmotionAnalysisSystem

app = FastAPI(
    title="Ukrainian Emotion Analysis API",
    description="API for analyzing emotions and sentiment in Ukrainian text",
    version="1.0.0"
)

# Initialize the emotion analysis system
analyzer = EmotionAnalysisSystem()

class TextInput(BaseModel):
    text: str
    use_ukrainian: Optional[bool] = True
    threshold: Optional[float] = 0.2

class BatchTextInput(BaseModel):
    texts: List[str]
    use_ukrainian: Optional[bool] = True
    threshold: Optional[float] = 0.2

@app.post("/analyze")
async def analyze_text(input_data: TextInput):
    """
    Analyze a single text for both sentiment and emotions.
    """
    try:
        result = analyzer.analyze(input_data.text)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch(input_data: BatchTextInput):
    """
    Analyze multiple texts in batch.
    """
    try:
        results = analyzer.analyze_batch(input_data.texts)
        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/sentiment")
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze only the sentiment of the text.
    """
    try:
        sentiment = analyzer.sentiment_analyzer.analyze(input_data.text)
        scores = analyzer.sentiment_analyzer.get_confidence_scores(input_data.text)
        return {
            "status": "success",
            "data": {
                "sentiment": sentiment,
                "confidence_scores": scores
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/emotions")
async def analyze_emotions(input_data: TextInput):
    """
    Analyze only the emotions in the text.
    """
    try:
        emotions = analyzer.emotion_analyzer.analyze(
            input_data.text,
            use_ukrainian=input_data.use_ukrainian
        )
        scores = analyzer.emotion_analyzer.get_confidence_scores(input_data.text)
        mixture = analyzer.emotion_analyzer.get_emotion_mixture(
            input_data.text,
            threshold=input_data.threshold
        )
        return {
            "status": "success",
            "data": {
                "dominant_emotion": emotions,
                "confidence_scores": scores,
                "emotion_mixture": mixture
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 