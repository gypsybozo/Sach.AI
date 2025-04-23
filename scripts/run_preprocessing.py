import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from unbiased_news_gen.data_ingestion import loader
from unbiased_news_gen.preprocessing import cleaner
from unbiased_news_gen.utils import helpers
from unbiased_news_gen import config
import logging
import pandas as pd # Import pandas
from tqdm import tqdm # Import tqdm

# Initialize tqdm to work with pandas apply
tqdm.pandas(desc="Preprocessing Articles")

def main():
    logging.info("--- Starting Preprocessing Pipeline ---")

    # 1. Load Data
    news_df = loader.load_news_data()
    if news_df is None:
        logging.error("Failed to load news data. Exiting.")
        return
    
    # --- Add this after news_df = loader.load_news_data() ---
    logging.warning("--- PROCESSING ONLY A SUBSET (100 rows) FOR TESTING ---")
    news_df = news_df.head(1000).copy()
    # ---------------------------------------------------------

    # 2. Preprocess Text Data
    logging.info(f"Starting text preprocessing for column '{'articles'}'...")
    if 'articles' not in news_df.columns:
        logging.error(f"Column '{'articles'}' not found in DataFrame.")
        return

    # Ensure the text column is string type, fill NaNs with empty string
    news_df['articles'] = news_df['articles'].astype(str).fillna('')

    # Apply preprocessing using progress_apply
    # NOTE: cleaner.preprocess_text needs to be adjusted to handle Series apply efficiently
    # Let's adjust the cleaner function slightly or apply row-wise (slower but simpler for now)
    # We will apply preprocess_text directly using progress_apply

    # Apply basic cleaning first efficiently
    logging.info("Applying basic text cleaning...")
    news_df['cleaned_text'] = news_df['articles'].progress_apply(cleaner.basic_text_cleaning)

    # Apply spaCy processing with progress bar
    # This is still the slow part
    logging.info("Applying spaCy NLP preprocessing (lemmatization, etc.)...")
    # Make sure preprocess_text uses the 'cleaned_text' now
    processed_results = news_df['cleaned_text'].progress_apply(lambda x: cleaner.preprocess_text(x))

    # Separate the results back into columns
    news_df['processed_tokens'] = processed_results.apply(lambda x: x) # x is already the list of tokens
    news_df['processed_text'] = news_df['processed_tokens'].apply(lambda tokens: ' '.join(tokens))

    # Drop the intermediate cleaned_text column if desired
    # news_df = news_df.drop(columns=['cleaned_text'])

    logging.info("Finished text preprocessing.")

    # 3. Save Processed Data
    helpers.save_csv(news_df, config.PREPROCESSED_ARTICLES_CSV)

    logging.info("--- Preprocessing Pipeline Finished ---")

if __name__ == "__main__":
    main()