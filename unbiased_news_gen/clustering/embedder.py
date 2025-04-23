from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer # Import SentenceTransformer
import logging
from unbiased_news_gen import config
import numpy as np
import torch # Import torch to check for GPU

# Check for GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {DEVICE} for transformer embeddings.")

# Load sentence transformer model once
if config.EMBEDDING_METHOD == 'transformer':
    try:
        logging.info(f"Loading SentenceTransformer model: {config.SENTENCE_TRANSFORMER_MODEL}")
        # Load the model onto the specified device
        transformer_model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL, device=DEVICE)
        logging.info(f"SentenceTransformer model loaded successfully onto {DEVICE}.")
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model '{config.SENTENCE_TRANSFORMER_MODEL}': {e}")
        transformer_model = None
else:
    transformer_model = None

def get_tfidf_embeddings(texts):
    # (Keep the existing get_tfidf_embeddings function as is)
    logging.info("Generating TF-IDF embeddings...")
    if not texts:
        return None, None
    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(texts)
        logging.info(f"Generated TF-IDF embeddings. Shape: {embeddings.shape}")
        return embeddings, vectorizer
    except Exception as e:
        logging.error(f"Error generating TF-IDF embeddings: {e}")
        return None, None


def get_transformer_embeddings(texts):
    """Generates embeddings using a pre-trained transformer model."""
    logging.info(f"Generating Transformer embeddings using {config.SENTENCE_TRANSFORMER_MODEL} on {DEVICE}...")
    if transformer_model is None:
        logging.error("Transformer model not loaded.")
        return None
    if not texts:
        return None

    try:
        # Ensure texts is a list of strings
        texts = list(texts)
        # Encode the texts. batch_size can be tuned based on GPU memory.
        embeddings = transformer_model.encode(texts,
                                              batch_size=32, # Adjust based on GPU memory
                                              show_progress_bar=True,
                                              device=DEVICE) # Explicitly specify device
        logging.info(f"Generated Transformer embeddings. Shape: {embeddings.shape}")
        # Embeddings are numpy arrays
        return embeddings
    except Exception as e:
        logging.error(f"Error generating Transformer embeddings: {e}")
        return None


def get_embeddings(texts, method=config.EMBEDDING_METHOD):
    """Dispatcher function to get embeddings based on the chosen method."""
    if method == 'tfidf':
        # TF-IDF returns a sparse matrix
        embeddings, _ = get_tfidf_embeddings(texts) # We only need the embeddings here
        return embeddings
    elif method == 'transformer':
        # Transformer models return dense numpy arrays
        return get_transformer_embeddings(texts)
    else:
        logging.error(f"Unknown embedding method: {method}")
        return None
    
def generate_entity_embeddings(entity_definitions_df, text_column='Wikipedia Content', entity_column='Entity'):
    """Generates embeddings for each entity based on its description."""
    logging.info(f"Generating embeddings for entity definitions using {config.SENTENCE_TRANSFORMER_MODEL}...")
    if transformer_model is None:
        logging.error("Transformer model not loaded.")
        return None
    if entity_definitions_df is None or text_column not in entity_definitions_df.columns or entity_column not in entity_definitions_df.columns:
         logging.error("Invalid entity definitions DataFrame provided.")
         return None

    # Ensure text is string
    entity_definitions_df[text_column] = entity_definitions_df[text_column].astype(str).fillna('')
    descriptions = entity_definitions_df[text_column].tolist()
    entity_names = entity_definitions_df[entity_column].tolist()

    try:
        embeddings = transformer_model.encode(descriptions,
                                              batch_size=32,
                                              show_progress_bar=True,
                                              device=DEVICE)
        entity_embedding_map = {name: emb for name, emb in zip(entity_names, embeddings)}
        logging.info(f"Generated embeddings for {len(entity_embedding_map)} entities.")
        # Save this map (e.g., using pickle or another format) for later use
        # helpers.save_pickle(entity_embedding_map, 'path/to/entity_embeddings.pkl')
        return entity_embedding_map
    except Exception as e:
        logging.error(f"Error generating entity embeddings: {e}")
        return None