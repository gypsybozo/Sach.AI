import spacy
from nltk.corpus import stopwords
import re
import logging
from unbiased_news_gen import config

# Load spaCy model once
try:
    nlp = spacy.load(config.SPACY_MODEL)
except OSError:
    logging.error(f"SpaCy model '{config.SPACY_MODEL}' not found. Please download it: python -m spacy download {config.SPACY_MODEL}")
    nlp = None

# Load stopwords once
stop_words = set(stopwords.words('english'))
# TODO: Consider adding custom Indian context stop words if needed

def basic_text_cleaning(text):
    """Performs basic cleaning: lowercase, removes punctuation, numbers, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, lemmatize=True, remove_stopwords=True):
    """Tokenizes, optionally removes stopwords and lemmatizes."""
    if nlp is None:
        logging.error("SpaCy model not loaded. Cannot preprocess text.")
        return [] # Return empty list or handle error as appropriate

    # Apply basic cleaning first
    cleaned_text = basic_text_cleaning(text)

    # Process text with spaCy
    doc = nlp(cleaned_text)
    processed_tokens = []

    for token in doc:
        # Filter out stop words if required
        if remove_stopwords and token.text in stop_words:
            continue
        # Filter out short tokens or tokens that became empty after cleaning
        if len(token.text.strip()) == 0:
            continue

        # Lemmatize if required, otherwise use the token text
        token_to_add = token.lemma_ if lemmatize else token.text
        processed_tokens.append(token_to_add)

    return processed_tokens

def preprocess_text_column(df, text_column='articles'):
    """Applies preprocessing to a DataFrame column."""
    logging.info(f"Starting text preprocessing for column '{text_column}'...")
    if text_column not in df.columns:
        logging.error(f"Column '{text_column}' not found in DataFrame.")
        return df

    # Ensure the text column is string type, fill NaNs with empty string
    df[text_column] = df[text_column].astype(str).fillna('')

    # Apply preprocessing
    # Note: This can be slow for large datasets. Consider using df.progress_apply with tqdm
    # or parallel processing libraries like pandarallel or dask.
    df['processed_tokens'] = df[text_column].apply(preprocess_text)
    df['processed_text'] = df['processed_tokens'].apply(lambda tokens: ' '.join(tokens)) # Join tokens back for some models

    logging.info("Finished text preprocessing.")
    return df