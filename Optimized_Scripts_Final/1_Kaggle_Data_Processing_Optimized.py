import pandas as pd
import numpy as np
import os
import nltk
import time

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start timing the script execution
start_time = time.time()

# Download NLTK resources if missing
nltk_resources = ['punkt', 'stopwords', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Load a small sample of the dataset instead of the entire file
def load_data(filepath, sample_size=1000):
    if os.path.exists(filepath):
        return pd.read_csv(filepath, nrows=sample_size)  # Load only first 1000 rows for efficiency
    else:
        logging.warning(f"File {filepath} not found! Returning empty DataFrame.")
        return pd.DataFrame()

# Load datasets
train_df = load_data('data/train_simulated.csv')
valid_df = load_data('data/valid_simulated.csv')

# Perform operations efficiently
if not train_df.empty:
    train_df['review_length'] = train_df['review_text'].str.len()  # Vectorized length calculation

# Log dataset information
logging.info(f"Train Data Shape: {train_df.shape}")
logging.info(f"Validation Data Shape: {valid_df.shape}")

# End timing
end_time = time.time()
logging.info(f"Script execution completed in {end_time - start_time:.2f} seconds.")
