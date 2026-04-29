# src/main.py

"""
--- Financial Product Complaint Classification and Summarization ---

Description
In today’s financial landscape, customer complaints offer critical insight into service gaps and operational inefficiencies. Automatically categorizing these complaints into product-specific segments—such as credit reporting, student loans, or money transfers—helps organizations streamline case handling and direct issues to the appropriate teams. By applying Generative AI for text classification and summarization, institutions can better interpret customer concerns and accelerate response workflows. Additionally, generating concise summaries of long complaints enables support teams to quickly understand the core issue without manually parsing lengthy narratives.*

Objective
This project demonstrates how Generative AI techniques can be applied to enhance both the classification and summarization of financial customer complaints. Specifically, it focuses on:*

1. **Text-to-Label Classification:** *Using Zero-shot and Few-shot prompting strategies to assign customer complaints to their correct product categories without requiring traditional supervised training datasets.*

2. **Text-to-Text Summarization:** *Applying Zero-shot prompting to produce clear, concise summaries that help support teams rapidly interpret customer issues.*

Conclusion
By completing this project, you will gain hands-on experience developing LLM-driven solutions for text classification and summarization. These capabilities enable financial institutions to automate key aspects of the complaint triage process—resulting in faster routing, more accurate responses, improved customer satisfaction, and enhanced regulatory compliance. The techniques demonstrated here also provide transferable skills applicable across a broad range of real-world NLP and enterprise automation scenarios.*

"""

# This part of code will skip all the un-necessary warnings which can occur during the execution of this project.
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

# Basic Imports for Libraries
try:
    from IPython.display import display
except ImportError:
    display = print

from tqdm import tqdm
import json
import re
import torch
import evaluate
import locale
import random
import time
import math

# Vendor Imports
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Local Imports
from src.config import *
from src.eda import *

from src.preprocessing import load_data
from src.utils import *
from src.modeling import *

# --- Start Program --- #

"""
Command lines:
>_ ./venv/bin/python main.py 
run everything in main.py

>_ ./venv/bin/python main.py -- eda
Run the exploratory data analysis (EDA) pipeline to understand the dataset and prepare it for modeling.
 
>_ ./venv/bin/python main.py -- seed
Add seeder data to dataset to create a few-shot prompt for the text classification task.
"""


# --- Run Pipelines --- #

def run_eda_pipeline(seed_data: bool = False):
    # Load data and show overview.
    df = load_data()

    # Load models
    show_overview(df)


# Run the main pipeline to execute the text classification and summarization tasks.
def run_main_pipeline(seed_data: bool = False):
    # Load data

    df = load_data()

    # Randomly select 30 rows as test data.
    random_data = df.sample(n=config.TEST_DATA_SIZE, random_state=config.SEED)


    # Importing Libraries and Mistral Model
    # Load and create an instance of the Llama c++ model.
    lcpp_llm = llama()

    # Text to Label generation

    """
    Define the Prompt Template, System Message, generate_prompt
    
    # - Define a **system message** as a string and assign it to the variable system_message to generate product class.
    # - Create a **zero shot prompt template** that incorporates the system message and user input.
    # - Define **generate_prompt** function that takes both the system_message and user_input as arguments and formats them into a prompt template
    #
    #
    # Write a Python function called **generate_mistral_response** that takes a single parameter, narrative, which represents the user's complain. Inside the function, you should perform the following tasks:
    #
    #
    # - **Combine the system_message and narrative to create a prompt string using generate_prompt function.**
    #
    # *Generate a response from the Mistral model using the lcpp_llm instance with the following parameters:*
    #
    # - prompt should be the combined prompt string.
    # - max_tokens should be set to 1200.
    # - temperature should be set to 0.
    # - top_p should be set to 0.95.
    # - repeat_penalty should be set to 1.2.
    # - top_k should be set to 50.
    # - stop should be set as a list containing '/s'.
    # - echo should be set to False.
    # Extract and return the response text from the generated response.
    #
    # Don't forget to provide a value for the system_message variable before using it in the function.
    """

    # --- Zero Shot Prompting for Text Classification ---
    run_zero_shot_text_classification()

    # --- Few Shot Prompting for Text to Label Classification ---
    run_few_shot_text_classification()

    # --- Text to Text generation ---
    run_zero_shot_text_summarization()

    # When evaluating Text-To-Text Summarization with the B.E.R.T Score we get a result of 0.320.
    #
    # However, if we add the parameter <code>model_type='distilbert-base-uncased'</code> it jumps to 0.515! The low score is because a zero-shot prompt returns a summary that differs from my gold-standard summaries in the data set. A score of 0.8 would be better.

    # --- END OF PIPELINE PROGRAM ---


if __name__ == "__main__":

    # @link https://docs.python.org/3/library/argparse.html
    # Check for arguments to determine which processes to run.  Check for 'seed', 'eda' or no arguments to run the full pipeline.
    parser = argparse.ArgumentParser(
        prog='Financial Product Complaint Classification and Summarization',
        description='Analyzes financial customer complaints using Generative AI for text classification and summarization.',
        epilog='Example usage: python main.py --eda to run only the EDA pipeline.')

    parser.add_argument('--eda', action='store_true', help='Run the EDA pipeline')
    parser.add_argument('--seed', action='store_true', help='Add seeder data')
    parser.add_argument('filename', nargs='?', default=None)           # positional argument
    parser.add_argument('-c', '--count')      # option that takes a value
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag

    args = parser.parse_args()

    # Run the EDA pipeline to understand the dataset and prepare it for modeling.
    if args.eda:
        run_eda_pipeline()
    else:
        # Run the main pipeline to execute the text classification and summarization tasks.
        run_main_pipeline()

# --- End Program --- #
