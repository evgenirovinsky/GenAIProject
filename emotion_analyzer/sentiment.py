from typing import Dict, List, Union, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

class SentimentAnalyzer:
    """
    Sentiment analyzer using a pre-trained Ukrainian BERT model.
    Classifies text into positive, negative, or neutral sentiment.
    """

    def __init__(self, device: str = None, token: Optional[str] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            device: The device to run the model on ('cuda' or 'cpu').
                   If None, will use CUDA if available, otherwise CPU.
            token: Hugging Face API token for accessing the model.
                  If None, will try to use the token from environment variable HF_TOKEN.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        # Using a Russian/Ukrainian BERT model fine-tuned for emotion detection
        self.model_name = "cointegrated/rubert-tiny2-cedr-emotion-detection"
        
        # Get token from parameter or environment variable
        self.token = token or os.getenv('HF_TOKEN')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            token=self.token
        )
        self.model.to(device)
        self.model.eval()

        # Actual model labels
        self.model_labels = list(self.model.config.label2id.keys())
        # Map model labels to sentiment categories
        self.label_map = {
            'joy': 'positive',
            'sadness': 'negative',
            'anger': 'negative',
            'fear': 'negative',
            'no_emotion': 'neutral',
            'surprise': 'neutral'
        }

    def analyze(self, text: str) -> str:
        """
        Analyze the sentiment of a single text.
        
        Args:
            text: The input text to analyze
            
        Returns:
            The predicted sentiment (positive/negative/neutral)
        """
        try:
            # Get model label scores (not aggregated sentiment scores)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)[0]
            emotion_scores = {
                label: score.item()
                for label, score in zip(self.model_labels, scores)
            }
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            mapped_sentiment = self.label_map.get(dominant_emotion, 'neutral')

            # Improved mapping: if positive and negative scores are close, set to neutral
            pos_score = emotion_scores.get('joy', 0.0)
            neg_score = emotion_scores.get('sadness', 0.0) + emotion_scores.get('anger', 0.0) + emotion_scores.get('fear', 0.0)
            neu_score = emotion_scores.get('no_emotion', 0.0) + emotion_scores.get('surprise', 0.0)
            if abs(pos_score - neg_score) < 0.15 and (pos_score > 0.2 or neg_score > 0.2):
                mapped_sentiment = 'neutral'

            # Lexicon-based adjustment: upgrade to positive if strong positive words are present
            positive_words = [
                'радість', 'гордість', 'вдячний', 'успіх', 'завершив', 'полегшення', 'щасливий', 'задоволення', 'святкувати', 'досягнення', 'команда', 'підтримка', 'вдячність', 'щастя', 'любов', 'натхнення'
            ]
            text_lower = text.lower()
            if mapped_sentiment == 'negative':
                for word in positive_words:
                    if word in text_lower:
                        mapped_sentiment = 'positive'
                        break
            return mapped_sentiment
        except Exception as e:
            return 'neutral'

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

        # Convert to dictionary with emotion labels
        emotion_scores = {
            label: score.item()
            for label, score in zip(self.model_labels, scores)
        }
        
        # Aggregate emotions into sentiment categories
        sentiment_scores = {
            'positive': emotion_scores.get('joy', 0.0),
            'negative': emotion_scores.get('sadness', 0.0) + emotion_scores.get('anger', 0.0) + emotion_scores.get('fear', 0.0),
            'neutral': emotion_scores.get('no_emotion', 0.0) + emotion_scores.get('surprise', 0.0)
        }
        
        # Normalize scores
        total = sum(sentiment_scores.values())
        if total > 0:
            sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
        
        return sentiment_scores

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