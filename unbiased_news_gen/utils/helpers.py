import pandas as pd
import os
import logging
import pickle
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Add these functions to helpers.py
def save_pickle(data, file_path):
    """Saves Python data to a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Successfully saved data to {os.path.basename(file_path)}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")

def load_pickle(file_path):
    """Loads Python data from a pickle file."""
    if not os.path.exists(file_path):
        logging.error(f"Pickle file not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded data from {os.path.basename(file_path)}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def load_csv(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {os.path.basename(file_path)}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def save_csv(df, file_path):
    """Saves a pandas DataFrame to a CSV file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully saved data to {os.path.basename(file_path)}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")

# Add other potential helpers here (e.g., JSON loading/saving)