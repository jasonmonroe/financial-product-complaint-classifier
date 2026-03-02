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

import config


# Returns shuffle data with entire row set.
def shuffle_data(df):
    return df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

# Returns sample data of x size from input data.
def create_sample_data(size: int, df):
    return df.sample(n=size, random_state=config.SEED)

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
        pattern1 = r'(' + label_pattern + ')'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group()
        else:
            return ''