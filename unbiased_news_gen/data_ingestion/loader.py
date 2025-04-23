from unbiased_news_gen.utils import helpers
from unbiased_news_gen import config
import pandas as pd # Ensure pandas is imported
import logging

def load_news_data():
    """Loads the Indian Express news dataset."""
    logging.info("Loading Indian Express news data...")
    df = helpers.load_csv(config.INDIAN_EXPRESS_CSV)
    # Basic validation
    if df is not None and 'articles' not in df.columns:
         logging.error("Required 'articles' column not found in news data.")
         return None
    if df is not None:
        # Drop rows where article text is missing
        df.dropna(subset=['articles'], inplace=True)
        logging.info(f"Dropped rows with missing articles. New shape: {df.shape}")
    return df


def load_entity_definitions():
    """Loads the custom entity definitions and creates a lookup dictionary."""
    logging.info("Loading custom entity definitions...")
    df = helpers.load_csv(config.ENTITY_DEFINITIONS_CSV)
    if df is None:
        return None, None # Return None for both dataframe and lookup dict

    # Basic validation
    required_columns = ['Entity', 'Wikipedia Content', 'Category']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Required columns ({required_columns}) not found in entity definitions.")
        return df, None # Return dataframe but None for lookup

    # Create a lookup dictionary: lowercase entity name -> {'category': Category, 'content': Wikipedia Content}
    entity_lookup = {}
    for _, row in df.iterrows():
        entity_name_lower = str(row['Entity']).lower()
        entity_lookup[entity_name_lower] = {
            'category': str(row['Category']),
            'content': str(row['Wikipedia Content'])
            # Add other relevant fields from the CSV if needed
        }
    logging.info(f"Created lookup for {len(entity_lookup)} unique lowercase entity definitions.")
    return df, entity_lookup # Return both the original DataFrame and the lookup dict

# TODO: Add functions for scraping or API clients if needed later