import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from unbiased_news_gen.utils import helpers
from unbiased_news_gen import config
from unbiased_news_gen.bias_detection import ner, sentiment, lexical, scorer
import pandas as pd
import logging
from tqdm import tqdm # For progress bar

tqdm.pandas() # Enable progress_apply for pandas

def analyze_article_bias(row):
    """Analyzes a single article (row) for bias."""
    text = row['articles'] # Use original text for analysis
    processed_tokens = row['processed_tokens'] # Use tokens for lexical analysis

    # 1. NER
    entities = ner.extract_entities_enhanced(text)

    # 2. Sentiment (Overall and Entity-Specific Placeholder)
    overall_sentiment = sentiment.get_sentiment_vader(text)
    
    entity_sentiments = sentiment.get_entity_sentiment_context(text, entities)

    # 3. Lexical Bias
    lexical_results = lexical.detect_lexical_bias(processed_tokens)

    # 4. Scoring
    bias_scores = scorer.calculate_bias_score_refined(overall_sentiment, lexical_results, entity_sentiments)

    # Combine results - return a dictionary or Series
    results = {
        'entities': entities,
        'overall_sentiment': overall_sentiment,
        'entity_sentiments': entity_sentiments, # Placeholder results
        'lexical_analysis': lexical_results,
        **bias_scores # Unpack the dictionary from scorer
    }
    return pd.Series(results)


def main():
    logging.info("--- Starting Bias Detection Pipeline ---")

    # 1. Load Preprocessed Data
    df = helpers.load_csv(config.PREPROCESSED_ARTICLES_CSV)
    if df is None:
        logging.error("Failed to load preprocessed data. Run preprocessing script first.")
        return
    if 'articles' not in df.columns or 'processed_tokens' not in df.columns:
        logging.error("Required columns ('articles', 'processed_tokens') not found in preprocessed data.")
        return

    # Limit rows for faster testing initially
    # df = df.head(100).copy()
    # logging.warning("Processing only the first 100 rows for testing.")


    # 2. Apply Bias Analysis to each article
    logging.info(f"Applying bias analysis to {len(df)} articles...")
    # Use progress_apply for visual feedback
    bias_analysis_results = df.progress_apply(analyze_article_bias, axis=1)

    # 3. Combine results with original DataFrame
    output_df = pd.concat([df, bias_analysis_results], axis=1)

    # Select relevant columns for saving (optional)
    # columns_to_save = ['article_id', 'headline', 'date', 'final_bias_score', 'entities', 'lexical_terms_found', ...]
    # output_df = output_df[columns_to_save]

    # 4. Save Results
    helpers.save_csv(output_df, config.BIAS_ANALYSIS_CSV)

    logging.info("--- Bias Detection Pipeline Finished ---")

if __name__ == "__main__":
    main()