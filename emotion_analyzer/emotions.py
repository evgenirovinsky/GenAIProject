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
        # Using a Russian/Ukrainian BERT model fine-tuned for emotion detection
        self.model_name = "cointegrated/rubert-tiny2-cedr-emotion-detection"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

        # Ukrainian emotion labels
        self.emotion_map = {
            'joy': 'радість',
            'sadness': 'сум',
            'anger': 'злість',
            'fear': 'страх',
            'no_emotion': 'нейтральність',
            'surprise': 'здивування'
        }

        # Additional emotion mappings based on sentiment intensity
        self.emotion_intensity_map = {
            'joy': {
                'high': 'захоплення',
                'medium': 'радість',
                'low': 'задоволення'
            },
            'sadness': {
                'high': 'горе',
                'medium': 'сум',
                'low': 'розчарування'
            },
            'anger': {
                'high': 'лють',
                'medium': 'злість',
                'low': 'роздратування'
            },
            'fear': {
                'high': 'жах',
                'medium': 'страх',
                'low': 'тривога'
            },
            'no_emotion': {
                'high': 'спокій',
                'medium': 'нейтральність',
                'low': 'байдужість'
            },
            'surprise': {
                'high': 'шок',
                'medium': 'здивування',
                'low': 'подив'
            }
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
            # Get the base emotion in Ukrainian
            base_emotion = self.emotion_map[dominant_emotion]
            
            # Get the intensity of the emotion
            score = scores[dominant_emotion]
            if score > 0.8:
                intensity = 'high'
            elif score > 0.5:
                intensity = 'medium'
            else:
                intensity = 'low'
                
            # Return the detailed emotion based on intensity
            return self.emotion_intensity_map[dominant_emotion][intensity]
            
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

        # Convert to dictionary with base emotions
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