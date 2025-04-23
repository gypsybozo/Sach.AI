import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from unbiased_news_gen.utils import helpers
from unbiased_news_gen import config
from unbiased_news_gen.clustering import embedder, clusterer
import pandas as pd
import logging

def main():
    logging.info("--- Starting Clustering Pipeline ---")

    # 1. Load Bias Analysis Data (or preprocessed data if running separately)
    # We need the text column ('processed_text' or 'articles') for embedding
    df = helpers.load_csv(config.BIAS_ANALYSIS_CSV) # Assumes bias detection ran first
    if df is None:
        logging.warning("Bias analysis file not found, trying preprocessed data...")
        df = helpers.load_csv(config.PREPROCESSED_ARTICLES_CSV)
        if df is None:
             logging.error("Failed to load data for clustering. Run preprocessing/bias detection first.")
             return

    if 'processed_text' not in df.columns:
        logging.error("Required 'processed_text' column not found for embedding.")
        # Optional: Try 'articles' column if 'processed_text' is missing, but embeddings might differ
        if 'articles' in df.columns:
            logging.warning("Using 'articles' column for embedding as 'processed_text' is missing.")
            text_column = 'articles'
        else:
            return
    else:
         text_column = 'processed_text'


    # Ensure the text column is suitable (list of strings)
    texts_for_embedding = df[text_column].astype(str).tolist()

    # Limit rows for faster testing initially
    # texts_for_embedding = texts_for_embedding[:1000]
    # df = df.head(1000).copy()
    # logging.warning("Processing only the first 1000 rows for clustering testing.")


    # 2. Generate Embeddings
    logging.info(f"Using embedding method: {config.EMBEDDING_METHOD}")
    embeddings = embedder.get_embeddings(texts_for_embedding, method=config.EMBEDDING_METHOD)

    if embeddings is None:
        logging.error("Failed to generate embeddings. Exiting.")
        return

    # 3. Perform Clustering
    logging.info("Assigning clusters...")
    # Choose clustering method and parameters
    cluster_labels, cluster_params = clusterer.assign_clusters(embeddings, method=config.CLUSTER_METHOD)    # Or: cluster_labels = clusterer.assign_clusters(embeddings, method='dbscan', eps=0.7, min_samples=5)

    if cluster_labels is None:
        logging.error("Clustering failed.")
        return

    # 4. Add Cluster Labels and Info to DataFrame
    df['cluster_label'] = cluster_labels
    # Store the parameters used for reproducibility/info
    if config.CLUSTER_METHOD == 'kmeans':
        df['kmeans_k'] = cluster_params
    elif config.CLUSTER_METHOD == 'dbscan':
        df['dbscan_eps'] = cluster_params['eps']
        df['dbscan_min_samples'] = cluster_params['min_samples']

     # 5. Save Results
    helpers.save_csv(df, config.CLUSTERED_ARTICLES_CSV)


    logging.info("--- Clustering Pipeline Finished ---")

if __name__ == "__main__":
    main()