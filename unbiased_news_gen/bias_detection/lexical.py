from unbiased_news_gen import config
import logging

def detect_lexical_bias(tokens):
    """Detects predefined biased terms in a list of tokens."""
    if not isinstance(tokens, list):
        return {"score": 0.0, "terms_found": []}

    bias_score = 0.0
    terms_found = []
    # TODO: Improve matching (e.g., handle variations, multi-word phrases)
    text_lower = ' '.join(tokens).lower() # Join for easier substring search (simple approach)

    for term, score in config.LEXICAL_BIAS_TERMS.items():
        if term in text_lower: # Simple check
            bias_score += score
            terms_found.append({"term": term, "score": score})

    # Normalize score? (Optional, depends on scoring strategy)
    # E.g., normalize by text length or number of biased terms?

    return {"score": bias_score, "terms_found": terms_found}