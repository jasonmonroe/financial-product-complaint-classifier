# src/config.py

import random

# --- Constants ---
BERT_SCORE_MODEL = 'distilbert-base-uncased'

# This will be the seed value for generating random data.
# RANDOM_STATE_NUM = random.randint(1, 100)
SEED = 42
PRODUCT_LABELS = ['credit_card', 'retail_banking', 'credit_reporting', 'mortgages_and_loans', 'debt_collection']

# Number of rows of each product category (for training).
PRODUCT_SAMPLE_SIZE = 2

# Split of data for training and testing.
TEST_DATA_SIZE = 30 # Number of rows for testing.
TRAINING_DATA_SIZE = 50 # Number of rows for training.

# Files
CONTENT_PATH = '/content/drive/'
GOOGLE_DRIVE_PATH = 'MyDrive/Colab Notebooks/GA-NLP/project-01/'

csv_file_path = CONTENT_PATH + GOOGLE_DRIVE_PATH
csv_file_name = 'Complaints_classification.csv'

# Models

# Define Mistral Attributes
# @link https://mistral.ai/news/announcing-mistral-7b
# @link https://huggingface.co/mistralai/Mistral-7B-v0.1

MISTRAL_ATTRS = {
    'echo': False,
    'max_tokens': 1200,
    'repeat_penalty': 1.2,
    'stop_sequences': ['</s>'], # Note: notebook suggested '/s' but research suggested  '</s>'
    'temperature': 0,
    'top_k': 50,
    'top_p': 0.95,
}


# Define the attributes of the Llama c++ model
MODEL_ATTRS = {
    'batch_size': 512,
    'context_window': 4096,
    'cpu_cores': 2,
    'gpu_layers': 43,
    'repeat_penalty': 1.2,
    'temperature': 0,
    'top_p': 0.95
}

# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_BASENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL_PATH = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

