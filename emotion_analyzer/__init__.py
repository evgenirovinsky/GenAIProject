"""
Ukrainian Text Emotion Analysis System

This package provides tools for analyzing emotions and sentiment in Ukrainian text.
It combines sentiment analysis with detailed emotion detection using state-of-the-art
transformer models.

Main components:
- EmotionAnalysisSystem: Main class that combines sentiment and emotion analysis
- SentimentAnalyzer: Class for basic sentiment analysis
- EmotionAnalyzer: Class for detailed emotion detection

Example usage:
    from emotion_analyzer.analyzer import EmotionAnalysisSystem
    
    analyzer = EmotionAnalysisSystem()
    result = analyzer.analyze("Я дуже радий зустріти вас!")
    print(result)
"""

from .analyzer import EmotionAnalysisSystem
from .sentiment import SentimentAnalyzer
from .emotions import EmotionAnalyzer

__version__ = "1.0.0"
__all__ = ["EmotionAnalysisSystem", "SentimentAnalyzer", "EmotionAnalyzer"] 