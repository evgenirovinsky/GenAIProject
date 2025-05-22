from typing import Dict, List, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalyzer:
    """
    Sentiment analyzer using a pre-trained Ukrainian BERT model.
    Classifies text into positive, negative, or neutral sentiment.
    """

    def __init__(self, device: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            device: The device to run the model on ('cuda' or 'cpu').
                   If None, will use CUDA if available, otherwise CPU.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_name = "SkolkovoInstitute/rubert-base-cased-sentiment"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

        # Sentiment labels
        self.labels = ['negative', 'neutral', 'positive']

    def analyze(self, text: str) -> str:
        """
        Analyze the sentiment of a single text.
        
        Args:
            text: The input text to analyze
            
        Returns:
            The predicted sentiment (positive/negative/neutral)
        """
        scores = self.get_confidence_scores(text)
        return max(scores.items(), key=lambda x: x[1])[0]

    def get_confidence_scores(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for each sentiment class.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary mapping sentiment labels to their confidence scores
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
            for label, score in zip(self.labels, scores)
        }

    def analyze_batch(self, texts: List[str]) -> List[str]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of predicted sentiments
        """
        return [self.analyze(text) for text in texts]

    def get_sentiment_score(self, text: str) -> float:
        """
        Get a single sentiment score ranging from -1 (negative) to 1 (positive).
        
        Args:
            text: The input text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        scores = self.get_confidence_scores(text)
        # Convert to -1 to 1 scale using weighted average
        return (
            scores['positive'] - scores['negative']
        ) / (scores['positive'] + scores['negative'] + 1e-8) 