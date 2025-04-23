from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from unbiased_news_gen import config
from nltk.tokenize import sent_tokenize # Import sentence tokenizer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_vader(text):
    # (Keep the existing get_sentiment_vader function as is)
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    try:
        vs = analyzer.polarity_scores(text)
        return vs
    except Exception as e:
        logging.error(f"Error getting VADER sentiment: {e}")
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

def get_entity_sentiment_context(text, entities):
    """
    Calculates sentiment for entities based on the sentences they appear in.
    This is a basic form of Aspect-Based Sentiment Analysis.
    """
    entity_sentiments = []
    if not isinstance(text, str) or not text.strip() or not entities:
        return entity_sentiments

    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logging.error(f"NLTK sentence tokenization failed: {e}. Falling back to overall sentiment.")
        # Fallback: Assign overall sentiment to all entities if sentence tokenization fails
        overall_sentiment = get_sentiment_vader(text)['compound']
        for entity in entities:
             entity_sentiments.append({
                "entity_text": entity['text'],
                "label": entity.get('label', 'UNKNOWN'),
                "is_custom": entity.get('is_custom', False),
                "sentiment_score": overall_sentiment,
                "analysis_type": "overall_fallback",
                "context_sentences": []
             })
        return entity_sentiments

    for entity in entities:
        entity_text = entity['text']
        relevant_sentences = []
        sentence_scores = []

        # Find sentences containing the entity text (case-insensitive search)
        for sentence in sentences:
            if entity_text.lower() in sentence.lower():
                relevant_sentences.append(sentence)
                vs = get_sentiment_vader(sentence)
                sentence_scores.append(vs['compound'])

        # Calculate average sentiment from relevant sentences
        avg_score = 0.0
        if sentence_scores:
            avg_score = sum(sentence_scores) / len(sentence_scores)

        entity_sentiments.append({
            "entity_text": entity_text,
            "label": entity.get('label', 'UNKNOWN'),
            "is_custom": entity.get('is_custom', False),
            "category": entity.get('category', 'Unknown'), # Pass category along
            "sentiment_score": avg_score,
            "analysis_type": "sentence_context",
            "context_sentences": relevant_sentences
        })

    return entity_sentiments

# Replace the old function call in run_bias_detection.py
# from: entity_sentiments = sentiment.get_entity_sentiment(text, entities)
# to:   entity_sentiments = sentiment.get_entity_sentiment_context(text, entities)