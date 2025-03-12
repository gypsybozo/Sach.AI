import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load the BERT-based sentiment analysis model
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Download necessary NLP resources
nltk.download("stopwords")
nltk.download("punkt")

# Load the IndicNER model for Indian named entity recognition
model_name = "ai4bharat/IndicNER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Load the scraped articles
with open("news_articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Define stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Remove extra whitespace, newlines, and normalize punctuation."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text.lower()

def preprocess_text(text):
    """Tokenize, remove stopwords, and clean text."""
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

def extract_named_entities(text):
    """Extract named entities and merge subwords correctly."""
    ner_results = ner_pipeline(text[:512])  # Limit input to first 512 tokens

    entities = {}
    current_entity = []
    current_label = None

    for entity in ner_results:
        word = entity["word"]
        label = entity["entity"]

        # If it's a continuation of the same entity type, merge it
        if label.startswith("I-") and current_label == label:
            current_entity.append(word)
        else:
            # Store the previous entity if exists
            if current_entity:
                entity_name = " ".join(current_entity)
                entities[entity_name] = current_label

            # Start a new entity
            current_entity = [word]
            current_label = label

    # Store the last entity
    if current_entity:
        entity_name = " ".join(current_entity)
        entities[entity_name] = current_label

    return entities




def get_bert_sentiment(text):
    """Use BERT-based model to classify sentiment."""
    result = sentiment_pipeline(text[:512])  # Limit text to first 512 tokens
    return result[0]["label"], result[0]["score"]

# Apply preprocessing and sentiment analysis to all articles
for article in articles:
    article["content"] = clean_text(article["content"])
    article["summary"] = clean_text(article["summary"])
    article["content"] = preprocess_text(article["content"])
    article["summary"] = preprocess_text(article["summary"])
    article["entities"] = extract_named_entities(article["content"])
    
    # Apply BERT sentiment analysis
    label, score = get_bert_sentiment(article["content"])
    article["sentiment_label"] = label
    article["sentiment_score"] = score

# Save the cleaned and analyzed data
with open("new_cleaned_news_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=4)

print("Preprocessing complete! Cleaned articles saved to cleaned_news_articles.json")