# src/utils.py

# ---------------------------------
#  HELPER FUNCTIONS
# ---------------------------------
# Define helper functions to create sample data, create mistral response, clean mistral response.
# Define default row sizes for gold examples and test data.
#

import math
import re
import time
from src.config import SEED, PRODUCT_LABELS

import pandas as pd

# Returns shuffle data with entire row set.
def shuffle_data(df):
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Returns sample data of x size from input data.
def create_sample_data(size: int, df):
    return df.sample(n=size, random_state=SEED)

# Outputs banner for readability
def show_banner(title: str = ''):
    dashes = '-' * len(title)
    print(f"\n\n{title}\n{dashes}")

# Starts benchmark time.
def start_timer():
    return time.time()

# Ends benchmark time and returns formatted string of benchmark.
def output_timer(start_time: float, title: str):
    SECS_IN_MIN = 60
    MILLI_IN_SECS = 1000

    # Ends time and outputs the benchmark
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate minutes (the integer part of the total seconds divided by 60)
    minutes = int(elapsed_time // SECS_IN_MIN)

    # Calculate remaining seconds (the integer part of the total seconds modulo 60)
    seconds = int(elapsed_time % SECS_IN_MIN)

    # Calculate milliseconds (the fractional part multiplied by 1000 and rounded to an integer)
    milliseconds = int((elapsed_time - math.floor(elapsed_time)) * MILLI_IN_SECS)

    # Create the formatted string
    formatted_time = f"{minutes}m {seconds}s {milliseconds}ms"

    show_banner(title)
    print(f"Run Time: {formatted_time}.\n")

# Function looks for a category label and extracts it.
def extract_category(text):
    # Define the regex pattern to match "category:" or "Category:" followed by a word
    pattern = r'category:\s*(\w+)'  # The pattern itself remains the same

    # Use re.search with the re.IGNORECASE flag to make it case-insensitive
    match = re.search(pattern, text, re.IGNORECASE)

    # If a match is found, return the captured group, else return None
    if match:
        return match.group(1)
    else:
        label_pattern, _ = get_labels()
        pattern1 = r'(' + label_pattern + ')'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group()
        else:
            return ''

def get_labels():
    """
    Product Labels - these are the values in the product column in the dataset.
    Note: We will initialize it here but override once we load the actual data.
    """

    product_labels = PRODUCT_LABELS
    label_pattern = '|'.join(product_labels)
    labels_str = ', '.join(product_labels)

    return label_pattern, labels_str

def get_unique_product_labels(data):
    # Get unique product categories from the dataset.
    labels = data['product'].unique()
    label_pattern = '|'.join(labels)
    labels_str = ', '.join(labels)

    return labels, label_pattern, labels_str

# Create matches matrix to compare values of each column to gauge accuracy.
def create_match_results(new_data):
    # Count columns that match per row to compare which results match the actual product.
    total_rows = new_data.shape[0]
    print(f'Total Rows: {total_rows}')

    matches = {
        'product_v_response': {},
        'product_v_response_cleaned': {},
        'response_v_response_cleaned': {},
        'all_3': {}
    }

    # Count matches
    matches['product_v_response']['cnt'] = (new_data['product'] == new_data['mistral_response']).sum()
    matches['product_v_response_cleaned']['cnt'] = (new_data['product'] == new_data['mistral_response_cleaned']).sum()
    matches['response_v_response_cleaned']['cnt'] = (
            new_data['mistral_response'] == new_data['mistral_response_cleaned']).sum()
    matches['all_3']['cnt'] = ((new_data['product'] == new_data['mistral_response']) & (
            new_data['mistral_response'] == new_data['mistral_response_cleaned'])).sum()

    # Calculate match percentages
    for key in matches:
        matches[key]['pct'] = matches[key]['cnt'] / total_rows

    return matches


# Convert to DataFrame for display
def display_match_results(matches):
    # Define the human-readable labels
    match_labels = {
        'product_v_response': 'Product & Mistral Response',
        'product_v_response_cleaned': 'Product & Cleaned Mistral Response',
        'response_v_response_cleaned': 'Mistral Response & Cleaned Mistral Response',
        'all_3': 'Product & Mistral Response & Cleaned Mistral Response'
    }

    # Convert to DataFrame for display.
    matches_df = pd.DataFrame(matches).T.reset_index()
    matches_df.columns = ['Match Type', 'Count', 'Percentage']
    matches_df['Percentage'] = matches_df['Percentage'].apply(lambda x: f"{x:.2%}")

    # Apply the mapping to the 'Match Type' column.
    matches_df['Match Type'] = matches_df['Match Type'].map(match_labels)

    return matches_df


# Create a dataframe set of examples of each product category for training data.
# This will be used for few shot prompting.
def create_examples_df(data: pd.DataFrame, size: int, is_shuffle: bool = False) -> pd.DataFrame:
    """
    Creates a training set by sampling 'size' examples from each unique product category.
    """
    examples = {}
    labels, _, _ = get_unique_product_labels(data)

    for label in labels:
        reviews = data[data['product'] == label]

        # Sample x rows from each product.
        examples[label] = create_sample_data(size, reviews)

    labels_list = list(examples.values())

    # Add it to the examples dataframe set.
    examples_df = pd.concat(labels_list)

    # Shuffle the data one more time if flag is true
    if is_shuffle:
        examples_df = shuffle_data(examples_df)

    return examples_df
