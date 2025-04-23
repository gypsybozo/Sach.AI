import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Input files
INDIAN_EXPRESS_CSV = os.path.join(RAW_DATA_DIR, 'indian_express_news.csv')
ENTITY_DEFINITIONS_CSV = os.path.join(RAW_DATA_DIR, 'entities_definitions.csv')

# Output files (examples)
PREPROCESSED_ARTICLES_CSV = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_articles.csv')
BIAS_ANALYSIS_CSV = os.path.join(PROCESSED_DATA_DIR, 'bias_analysis_results.csv')
CLUSTERED_ARTICLES_CSV = os.path.join(PROCESSED_DATA_DIR, 'clustered_articles.csv')

# --- Model Settings ---
SPACY_MODEL = 'en_core_web_sm'
# Consider 'all-mpnet-base-v2' or 'paraphrase-multilingual-mpnet-base-v2' for broader coverage
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# --- NER & Entity Linking ---
# Threshold for fuzzy matching (0-100). Higher means stricter matching.
FUZZY_MATCH_THRESHOLD = 85 # Adjust based on testing

# --- Bias Detection Settings ---
# IMPORTANT: This list requires significant research for Indian context accuracy.
# Examples covering different potential bias areas. Scores are illustrative.
LEXICAL_BIAS_TERMS = {
    # Political (Example)
    "appeasement": -0.6, "vote bank": -0.5, "pseudo-secular": -0.6, "anti-national": -0.8,
    "masterstroke": 0.5, "visionary leader": 0.4, "strong government": 0.3, "failed government": -0.4,
    "jumla": -0.5, "tukde-tukde gang": -0.9,
    # Religious (Example)
    "radical islam": -0.7, "hindu extremist": -0.7, "forced conversion": -0.6, "majority appeasement": -0.5,
    "minority appeasement": -0.5, "love jihad": -0.8, "islamophobia": -0.7, "hinduphobia": -0.7,
    # Caste (Example)
    "caste discrimination": -0.7, "upper caste arrogance": -0.6, "dalit atrocities": -0.8,
    "reservation politics": -0.4,
    # Regional/Ethnic (Example)
    "illegal immigrant": -0.6, "separatist": -0.7, "mainlander": -0.3,
    # General Negative Loaded Language
    "regressive": -0.5, "divisive": -0.6, "controversial": -0.3, "so-called": -0.4,
    # Add many more based on research...
}

# --- Sentiment Analysis ---
# Method for Aspect-Based Sentiment (placeholder currently uses sentence context)
ABSA_METHOD = 'sentence_context' # Options: 'sentence_context', 'overall' (fallback)

# --- Scoring Weights (Tune these based on validation) ---
BIAS_SCORE_WEIGHTS = {
    'overall_sentiment': 0.10,  # Weight for overall article tone
    'lexical': 0.40,           # Weight for predefined biased words
    'entity_sentiment': {      # Weights for average sentiment towards custom entities BY CATEGORY
        'Political': 0.30,     # e.g., Weight for sentiment towards political entities
        'Religion': 0.60,     # e.g., Give more weight to sentiment towards religious groups
        'Caste': 0.60,         # e.g., Give more weight to sentiment towards caste groups
        'Region': 0.20,      # e.g., Less weight for sentiment towards states/regions (unless specific issue)
        'Gender': 0.50,        # e.g., Weight for sentiment towards gender entities
        'Other': 0.10          # Fallback weight if category not matched
    }
}

# --- Clustering Settings ---
# EMBEDDING_METHOD = 'tfidf' # Original
EMBEDDING_METHOD = 'transformer' # Switch to transformer
CLUSTER_METHOD = 'kmeans' # Options: 'kmeans', 'dbscan'
NUM_CLUSTERS_KMEANS = 15 # Example: Tune this using methods below
CLUSTER_TUNING_KMEANS = 'silhouette' # Options: 'silhouette', 'elbow', None
CLUSTER_RANGE_KMEANS = range(5, 30, 2) # Range of k to test for tuning
# DBSCAN parameters (need tuning if used)
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5