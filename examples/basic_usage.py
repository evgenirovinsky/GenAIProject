from emotion_analyzer.analyzer import EmotionAnalysisSystem

def main():
    # Initialize the emotion analysis system
    analyzer = EmotionAnalysisSystem()

    # Example texts in Ukrainian
    texts = [
        "Я дуже радий зустріти вас! Це чудовий день!",
        "Мені дуже сумно, що ви не прийшли на зустріч.",
        "Я здивований вашою реакцією на цю новину.",
        "Це просто нейтральне повідомлення без особливих емоцій."
    ]

    print("Analyzing individual texts:")
    print("-" * 50)
    
    for text in texts:
        print(f"\nText: {text}")
        
        # Get full analysis
        result = analyzer.analyze(text)
        print("\nFull Analysis:")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Dominant Emotion: {result['emotions']['dominant_emotion']}")
        print("Emotion Mixture:", ", ".join(result['emotions']['emotion_mixture']))
        
        # Get confidence scores
        sentiment_scores = analyzer.sentiment_analyzer.get_confidence_scores(text)
        emotion_scores = analyzer.emotion_analyzer.get_confidence_scores(text)
        
        print("\nConfidence Scores:")
        print("Sentiment scores:", sentiment_scores)
        print("Emotion scores:", emotion_scores)
        
        print("-" * 50)

    # Demonstrate batch processing
    print("\nBatch Processing Example:")
    print("-" * 50)
    
    batch_results = analyzer.analyze_batch(texts)
    for i, result in enumerate(batch_results):
        print(f"\nText {i+1}: {texts[i]}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Dominant Emotion: {result['emotions']['dominant_emotion']}")
        print("Emotion Mixture:", ", ".join(result['emotions']['emotion_mixture']))
        print("-" * 50)

if __name__ == "__main__":
    main() 