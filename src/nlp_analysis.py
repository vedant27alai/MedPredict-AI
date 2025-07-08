import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

def analyze_complaints(complaints):
    """Analyze complaints using sentiment analysis."""
    sia = SentimentIntensityAnalyzer()
    scores = []
    
    for complaint in complaints:
        # Placeholder for regional language translation
        score = sia.polarity_scores(complaint)['compound']
        normalized_score = (1 - score) / 2  # Normalize to [0, 1], 1 = negative
        scores.append(normalized_score)
    
    return scores