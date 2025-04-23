import logging
import numpy as np
from unbiased_news_gen import config # Import config for weights

def calculate_bias_score_refined(overall_sentiment, lexical_results, entity_sentiment_results):
    """
    Calculates a combined bias score using configurable category-specific
    weights for custom entities.
    """
    # Retrieve weights from config
    weights = config.BIAS_SCORE_WEIGHTS
    entity_cat_weights = weights.get('entity_sentiment', {})

    # 1. Overall Sentiment Component
    overall_sentiment_score = overall_sentiment.get('compound', 0.0)
    overall_weight = weights.get('overall_sentiment', 0.0)

    # 2. Lexical Bias Component
    lexical_score = lexical_results.get('score', 0.0)
    normalized_lexical_score = np.clip(lexical_score, -1.0, 1.0) # Simple normalization
    lexical_weight = weights.get('lexical', 0.0)

    # 3. Category-Specific Entity Sentiment Component
    sentiments_by_category = {}
    counts_by_category = {}
    num_custom_entities_scored = 0

    for res in entity_sentiment_results:
        if res.get('is_custom', False):
            num_custom_entities_scored += 1
            category = res.get('category', 'Other')
            # Use 'Other' as fallback category if not recognized
            if category not in entity_cat_weights:
                 category = 'Other'

            score = np.clip(res['sentiment_score'], -1.0, 1.0)

            if category not in sentiments_by_category:
                sentiments_by_category[category] = []
                counts_by_category[category] = 0
            sentiments_by_category[category].append(score)
            counts_by_category[category] += 1

    avg_sentiments_by_category = {}
    entity_sentiment_component = 0
    total_entity_weight_applied = 0

    logging.debug("--- Entity Sentiment Aggregation ---")
    for category, scores in sentiments_by_category.items():
        if scores:
            avg_score = np.mean(scores)
            avg_sentiments_by_category[category] = avg_score
            cat_weight = entity_cat_weights.get(category, 0.0) # Get weight for this category
            entity_sentiment_component += cat_weight * avg_score
            total_entity_weight_applied += abs(cat_weight) # Sum weights *actually used* based on found entities
            logging.debug(f"Category '{category}': AvgSent={avg_score:.3f}, Weight={cat_weight:.2f}, Count={counts_by_category[category]}")
        else:
            avg_sentiments_by_category[category] = 0.0
    logging.debug("--- End Entity Sentiment Aggregation ---")


    # 4. Combine Scores
    final_score_unnormalized = (overall_weight * overall_sentiment_score +
                                lexical_weight * normalized_lexical_score +
                                entity_sentiment_component)

    # 5. Normalize the final score based on max possible weighted score
    # Calculate sum of absolute values of *all possible* weights that *could* contribute
    max_possible_score = abs(overall_weight) + abs(lexical_weight) + sum(abs(w) for w in entity_cat_weights.values())

    if max_possible_score > 0:
        # Normalize by dividing by the sum of absolute weights
        final_score_normalized = final_score_unnormalized / max_possible_score
        final_score_normalized = np.clip(final_score_normalized, -1.0, 1.0) # Clip to ensure range
    else:
        final_score_normalized = 0.0 # Avoid division by zero

    logging.debug(
        f"Refined Bias Score Calc: OverallSent={overall_sentiment_score:.2f} (w={overall_weight:.2f}), "
        f"Lexical={normalized_lexical_score:.2f} (w={lexical_weight:.2f}), "
        f"EntityComponent={entity_sentiment_component:.3f} (MaxWeightSum={max_possible_score:.2f}) "
        f"-> Final Score={final_score_normalized:.4f}"
    )

    return {
        "final_bias_score": final_score_normalized,
        "overall_sentiment_compound": overall_sentiment_score,
        "lexical_bias_score": normalized_lexical_score,
        "avg_sentiments_by_category": avg_sentiments_by_category, # Avg score per category
        "num_custom_entities_scored": num_custom_entities_scored,
        "lexical_terms_found": lexical_results.get('terms_found', [])
    }