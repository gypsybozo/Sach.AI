import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from unbiased_news_gen.data_ingestion import loader
from unbiased_news_gen.clustering import embedder # Import embedder module
from unbiased_news_gen.utils import helpers
from unbiased_news_gen import config
import logging

ENTITY_EMBEDDINGS_PATH = os.path.join(config.PROCESSED_DATA_DIR, 'entity_embeddings.pkl')

def main():
    logging.info("--- Starting Entity Embedding Generation ---")

    # 1. Load Entity Definitions DataFrame
    entity_df, _ = loader.load_entity_definitions() # We only need the DataFrame here
    if entity_df is None:
        logging.error("Failed to load entity definitions DataFrame. Exiting.")
        return

    # Ensure the embedding method is 'transformer' for this script
    if config.EMBEDDING_METHOD != 'transformer':
        logging.error(f"Entity embedding generation requires EMBEDDING_METHOD='transformer' in config.py, but found '{config.EMBEDDING_METHOD}'. Exiting.")
        # Alternatively, you could force load the model here regardless of config, but using config is better.
        return


    # 2. Generate Embeddings
    # Pass the correct column names from your CSV
    entity_embeddings_map = embedder.generate_entity_embeddings(
        entity_df,
        entity_column='Entity',
        text_column='Wikipedia Content'
    )

    if entity_embeddings_map is None:
        logging.error("Failed to generate entity embeddings. Exiting.")
        return

    # 3. Save the Embeddings Map
    logging.info(f"Saving entity embeddings map to: {ENTITY_EMBEDDINGS_PATH}")
    helpers.save_pickle(entity_embeddings_map, ENTITY_EMBEDDINGS_PATH)

    logging.info("--- Entity Embedding Generation Finished ---")

if __name__ == "__main__":
    main()