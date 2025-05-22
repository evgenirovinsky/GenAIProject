from typing import Dict, List, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class EmotionAnalyzer:
    """
    Emotion analyzer using a multilingual RoBERTa model fine-tuned for emotion analysis.
    Detects emotions like joy, sadness, anger, fear, surprise, and neutral.
    """

    def __init__(self, device: str = None):
        """
        Initialize the emotion analyzer.
        
        Args:
            device: The device to run the model on ('cuda' or 'cpu').
                   If None, will use CUDA if available, otherwise CPU.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

        # English emotion labels and their Ukrainian translations
        self.emotion_map = {
            'joy': 'радість',
            'sadness': 'сум',
            'anger': 'злість',
            'fear': 'страх',
            'surprise': 'здивування',
            'neutral': 'нейтральність'
        }

    def analyze(self, text: str, use_ukrainian: bool = True) -> str:
        """
        Analyze the emotions in a single text.
        
        Args:
            text: The input text to analyze
            use_ukrainian: Whether to return emotions in Ukrainian
            
        Returns:
            The dominant emotion (in Ukrainian if use_ukrainian=True)
        """
        scores = self.get_confidence_scores(text)
        dominant_emotion = max(scores.items(), key=lambda x: x[1])[0]
        
        if use_ukrainian:
            return self.emotion_map[dominant_emotion]
        return dominant_emotion

    def get_confidence_scores(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for each emotion.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary mapping emotion labels to their confidence scores
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]

        # Convert to dictionary
        return {
            label: score.item()
            for label, score in zip(self.emotion_map.keys(), scores)
        }

    def analyze_batch(self, texts: List[str], use_ukrainian: bool = True) -> List[str]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of input texts to analyze
            use_ukrainian: Whether to return emotions in Ukrainian
            
        Returns:
            List of dominant emotions
        """
        return [self.analyze(text, use_ukrainian) for text in texts]

    def get_emotion_intensity(self, text: str) -> Dict[str, float]:
        """
        Get normalized emotion intensities (scores sum to 1).
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary mapping emotion labels to their normalized intensities
        """
        scores = self.get_confidence_scores(text)
        total = sum(scores.values())
        return {
            label: score / total
            for label, score in scores.items()
        }

    def get_emotion_mixture(self, text: str, threshold: float = 0.2) -> List[str]:
        """
        Get all emotions present in the text above a confidence threshold.
        
        Args:
            text: The input text to analyze
            threshold: Minimum confidence score for an emotion to be included
            
        Returns:
            List of emotions present in the text (in Ukrainian)
        """
        scores = self.get_confidence_scores(text)
        return [
            self.emotion_map[emotion]
            for emotion, score in scores.items()
            if score >= threshold
        ] 