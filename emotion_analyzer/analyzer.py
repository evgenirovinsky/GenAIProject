from typing import Dict, List, Union, Any, Optional
import torch
import os
from .sentiment import SentimentAnalyzer
from .emotions import EmotionAnalyzer

class EmotionAnalysisSystem:
    """
    Main class that combines sentiment and emotion analysis functionality.
    Uses both a sentiment analyzer and an emotion analyzer to provide
    comprehensive analysis of Ukrainian text.
    """

    def __init__(self, device: str = None, token: Optional[str] = None):
        """
        Initialize the emotion analysis system.
        
        Args:
            device: The device to run the models on ('cuda' or 'cpu').
                   If None, will use CUDA if available, otherwise CPU.
            token: Hugging Face API token for accessing the models.
                  If None, will try to use the token from environment variable HF_TOKEN.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.token = token or os.getenv('HF_TOKEN')
        
        # Initialize analyzers with token
        self.sentiment_analyzer = SentimentAnalyzer(device=device, token=self.token)
        self.emotion_analyzer = EmotionAnalyzer(device=device)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text for both sentiment and emotions.
        
        Args:
            text: The input text to analyze (in Ukrainian)
            
        Returns:
            Dictionary containing:
            - sentiment: The predicted sentiment (positive/negative/neutral)
            - emotions: Dictionary containing:
                - dominant_emotion: The emotion with highest confidence
                - emotion_mixture: List of emotions present in the text
        """
        sentiment = self.sentiment_analyzer.analyze(text)
        emotions = self.emotion_analyzer.analyze(text, use_ukrainian=True)
        emotion_mixture = self.emotion_analyzer.get_emotion_mixture(text)
        
        return {
            'sentiment': sentiment,
            'emotions': {
                'dominant_emotion': emotions,
                'emotion_mixture': emotion_mixture
            }
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of analysis results, one for each input text
        """
        return [self.analyze(text) for text in texts]

    def get_confidence_scores(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Get confidence scores for both sentiment and emotion predictions.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing confidence scores for both sentiment and emotions
        """
        sentiment_scores = self.sentiment_analyzer.get_confidence_scores(text)
        emotion_scores = self.emotion_analyzer.get_confidence_scores(text)
        
        return {
            'sentiment': sentiment_scores,
            'emotions': emotion_scores
        }

    def get_dominant_emotion(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Get the dominant emotion and its confidence score.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing the dominant emotion and its confidence score
        """
        scores = self.emotion_analyzer.get_confidence_scores(text)
        dominant_emotion = max(scores.items(), key=lambda x: x[1])
        
        return {
            'emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1]
        } 