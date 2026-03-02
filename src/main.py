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

from IPython.display import display

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

"""
Product Labels - these are the values in the product column in the dataset.
Note: We will initialize it here but override once we load the actual data.
"""


def get_labels():
    product_labels = config.PRODUCT_LABELS
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
    # Create reviews by extracting all rows by product. Then create examples for each product label.
    examples = {}
    for label in labels:
        reviews = data[data['product'] == label]

        # Sample x rows from each product.
        examples[label] = create_sample_data(size, reviews)

    labels_list = list(examples.values())

    # Add it to the examples dataframe set.
    examples_df = pd.concat(labels_list)

    # Shuffle the data one more time if flag is true
    if is_shuffle == True:
        examples_df = shuffle_data(examples_df)

    return examples_df




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

    #MODEL_PATH = init_model()

    # Randomly select 30 rows as test data.
    random_data = df.sample(n=config.TEST_DATA_SIZE, random_state=config.SEED)

    label_pattern, labels_str = get_labels()

    labels, label_pattern, labels_str = get_unique_product_labels(df)

    
    
    

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



    title = 'Zero-Shot Prompting for Text Classification'

    system_message = f""""""
    zero_shot_prompt_template = """"""


    # Set narrative data.
    df_sample = create_sample_data(config.TEST_DATA_SIZE, df)
    narratives = df_sample['narrative']

    # Get Zero-shot Mistral response
    # Takes about 15-20 seconds to run in Jupyter.
    start_time = start_timer()
    df_sample ['mistral_response'] = get_zero_shot_mistral_response(narratives)
    output_timer(start_time, title)
    
    print(df_sample['mistral_response'])

    # Get Mistral response and clean it.
    start_time = start_timer()
    df_sample['mistral_response_cleaned'] = clean_mistral_response(df_sample['mistral_response'])
    output_timer(start_time, title)
    print(df_sample['mistral_response_cleaned'])
    
    # Show overview of the sampled data
    show_overview(df_sample)

    # Calculate F1 score for 'product' and 'mistral_response' from scikit-learn.
    product = df_sample['product']
    mistral_response = df_sample['mistral_response']
    
    # Output F1 Mistral score.
    f1_mistral_response_score = f1_score(product, mistral_response, average='micro')

    # Zero Shot Prompt for Text Classification Results
    print(f'{title} Results')
    print(f'F1 Score: {f1_mistral_response_score}')

    # Calculate F1 score for product and mistral_response_cleaned.
    mistral_response_cleaned = df_sample['mistral_response_cleaned']
    f1_mistral_response_cleaned_score = f1_score(product, mistral_response_cleaned, average='micro')

    print(f'{title} Results')
    print(f'F1 Cleaned Score: {f1_mistral_response_cleaned_score}')


    # Calculate the delta between F1 Scores of Mistral response and the cleaned version.
    delta = f1_mistral_response_score - f1_mistral_response_cleaned_score
    delta = abs(delta)
    print(f'{title} Delta: {delta}')

    # Display clean table in Jupyter
    pd.set_option('display.max_colwidth', 256)
    matches = create_match_results(df_sample)
    df_matches = display_match_results(matches)
    display(df_matches)



    # --- Few Shot Prompting for Text Classification ---
    
    """
    Generate a set of gold examples by randomly selecting 10 instances of user_input and assistant_output from dataset 
    ensuring a balanced representation with 2 examples from each class.**

    """
    # Define title of exercise
    title = 'Few-Shot Text-to-Label Classification'

    # Create training set data for few shot prompting and create the training set by excluding examples.
    df_examples = create_examples_df(df, config.PRODUCT_SAMPLE_SIZE)
    df_gold_examples = df.drop(index=df_examples.index)

    # Convert examples to JSON
    columns_to_select = ['narrative', 'product']
    json_examples = df_examples[columns_to_select].to_json(orient='records')
    print(f'First record from JSON data: {json.loads(json_examples)[0]}')

    # Print the shapes of the datasets.
    # Note: Gold examples (also called "golden examples" or "ground truth examples") are high-quality, pre-verified
    # input-output pairs that serve as the "correct" or "ideal" examples for a specific task.
    print(f'Examples Set Shape: {df_examples.shape}')
    print(f'Gold Examples Shape: {df_gold_examples.shape}')





    """
    # - Define your **system_message**.
    # - Define **first_turn_template**, **example_template** and **prediction template**
    # - **create few shot prompt** using gold examples and system_message
    # - Randomly select 30 rows from test_df as test_data
    # - Create **mistral_response** with **mistral_response_cleaned** columns for this
    """

    system_message = f"""
    """
    print(f'few-shot: system_message: {system_message}')

    # Few Shot Templates for Mistral 7B

        # ----- First Turn -----
    first_turn_template = "<s>[INST]{system_message}\n\n{user_input}[/INST]{assistant_output}</s>"

    # ----- Examples -----
    examples_template = "<s>[INST]{user_input}[/INST]{assistant_output}</s>"

    # ----- Predictions -----
    prediction_template = "<s>[INST]{user_input}[/INST]"

    # Get Mistral response for few shot prompt.
    few_shot_prompt = create_few_shot_prompt(system_message, examples_df)
    print(few_shot_prompt)

    df_sample = create_sample_data(config.TEST_DATA_SIZE, df_gold_examples)
    narratives = df_sample['narrative']

    # This line may take a long time to process!
    start_time = start_timer()
    df_sample['mistral_response'] = get_few_shot_mistral_response(narratives)
    output_timer(start_time, title)


    # Few Shot Output
    # Calculate F1 score for 'product' and 'mistral_response'
    product = df_sample['product']
    mistral_response = df_sample['mistral_response']
    mistral_response_cleaned = df_sample['mistral_response_cleaned']

    # Get F1 score to output
    f1 = f1_score(product, mistral_response, average='micro')
    show_banner(title)

    # Results
    print(f'F1 Score: {f1}')

    # Few-Shot Prompt for Text Classification Results
    f1_cleaned = f1_score(product, mistral_response_cleaned, average='micro')


    # Few Shot Prompt for Text Classification Results
    print(f'Cleaned F1 Score: {f1_cleaned}')



    # --- Text to Text generation ---
    run_zero_shot_text_summarization()

    title = 'Zero-Shot Text Summarization'
    system_message = """
    You are an expert summarization tool for financial complaints. Your task is to provide a concise summary (1-3 sentences) of the complaint. The summary should focus on three key points:
    1. The main problem or core issue.
    2. The company or companies involved.
    3. The customer's desired outcome or the current status of the problem.
    
    If the complaint text is unclear or incomplete, leave the summary blank.
    **Only provide the summary.**  Do not provide any additional text.
    """

    print(f'system_message={system_message}')

    # Zero-Shot prompting for Text Summarization.
    zero_shot_prompt_template = "<s>[INST] {system_message}User Input: {user_input} [/INST]"

    # Create test data with gold examples for zero shot text-to-text summarization.
    df_gold_examples = create_sample_data(TEST_DATA_SIZE, df.copy())
    narratives = gold_examples['narrative']
    start_time = start_timer()
    df_gold_examples['mistral_response'] = get_zero_shot_mistral_response(narratives)
    output_timer(start_time, title)

    bert_scorer = evaluate.load('bertscore')

    # Get the score for text-to-text summarization.  For this summarization we will use the BERT score.
    start_time = start_timer()
    score = evaluate_score(df_gold_examples, bert_scorer, True)
    output_timer(start_time, title + ' BERT Score')

    print(f'BERT Score: {score}')






# In[ ]:


# # **Section 2: Text to Label generation**

# ### **Question 2: Zero-Shot Prompting for Text Classification**

# ##### **Q2.1: Define the Prompt Template, System Message, generate_prompt**



# In[ ]:


# Define title of exercise
title = 'Zero-Shot Prompting for Text Classification'

# In[ ]:


# Product Labels - these are the values in the product column in the dataset.
# Note: We will initialize it here but override once we load the actual data.
labels = ['credit_card', 'retail_banking', 'credit_reporting', 'mortgages_and_loans', 'debt_collection']
label_pattern = '|'.join(labels)
labels_str = ', '.join(labels)

# In[ ]:


# Instructions on labeling classifications as product categories.  The category will be used for extraction.
# Define the system message to only return a specific category name.
# Note: Reference: GANLP_Week3_MLS_Notebook_1.ipynb for insight.
system_message = f"""
System: You are an expert text classification model.
Your task is to classify a customer complaint into one of the following product categories: {labels_str}.
**Only return the category name** and nothing else.
If the product category name has a backslash in it, remove it.
"""

# In[ ]:


# Define the template used for prompting the labels.
zero_shot_prompt_template = """
<s>[INST] {system_message}

User Input: {user_input}
Category: [/INST]
"""

# **Refer the below reference image to upload a dataset file in the google colab.**

# ## Analyze the Classification Data

# In[1]:


CONTENT_PATH = '/content/drive/'
GOOGLE_DRIVE_PATH = 'MyDrive/Colab Notebooks/GA-NLP/project-01/'

csv_file_path = CONTENT_PATH + GOOGLE_DRIVE_PATH
csv_file_name = 'Complaints_classification.csv'  # <-- rename sample_data.csv

# In[ ]:


# Google Colab - Get data path
from google.colab import output, drive

drive.mount(CONTENT_PATH)

# In[ ]:


# Load a CSV File containing Dataset of 500 products, narrative and summary (summary of narrative)
csv_file = csv_file_path + csv_file_name

# Load data variable with CSV data.
data = pd.read_csv(csv_file)

# In[ ]:


# Randomly select 30 rows
new_data = data.sample(n=30, random_state=40)

# In[ ]:


# eda.py was here


# In[ ]:


# In[ ]:


# Confirm unique product categories and override previous declaration.
labels = data['product'].unique()
label_pattern = '|'.join(labels)
labels_str = ', '.join(labels)

# In[ ]:


# In[ ]:


# Randomly select 30 rows as test data.
new_data = create_sample_data(TEST_DATA_SIZE, data)

# Set narrative data.
narratives = new_data['narrative']

# In[ ]:


# Get Zero-shot Mistral response
# Takes about 15-20 seconds to run in Jupyter.
start_time = start_timer()
new_data['mistral_response'] = get_zero_shot_mistral_response(narratives)
output_timer(start_time, title)

# In[ ]:


new_data['mistral_response']

# In[ ]:


# In[ ]:


# Get Mistral response and clean it.
start_time = start_timer()
new_data['mistral_response_cleaned'] = clean_mistral_response(new_data['mistral_response'])
output_timer(start_time, title)
print(new_data['mistral_response_cleaned'])

# In[ ]:


new_data.head()

# ##### **Q2.3: Calculate the F1 score**

# In[ ]:


# Calculate F1 score for 'product' and 'mistral_response' from scikit-learn.
product = new_data['product']
mistral_response = new_data['mistral_response']

# Output F1 Mistral score.
f1_mistral_response_score = f1_score(product, mistral_response, average='micro')

# Zero Shot Prompt for Text Classification Results.
print(f'{title} Results')
print(f'F1 Score: {f1_mistral_response_score}')

# In[ ]:


# Calculate F1 score for product and mistral_response_cleaned.
mistral_response_cleaned = new_data['mistral_response_cleaned']

# Output F1 Mistral clean score.
f1_mistral_response_cleaned_score = f1_score(product, mistral_response_cleaned, average='micro')

print(f'{title} Results')
print(f'F1 Cleaned Score: {f1_mistral_response_cleaned_score}')

# In[ ]:


# Calculate the delta between F1 Scores of Mistral response and the cleaned version.
delta = f1_mistral_response_score - f1_mistral_response_cleaned_score
delta = abs(delta)
print(f'{title} Delta: {delta}')


# In[ ]:





# In[ ]:


# Display clean table in Jupyter
matches = create_match_results(new_data)
matches_df = display_match_results(matches)

pd.set_option('display.max_colwidth', 256)
from IPython.display import display

display(matches_df)

# ##### **Q2.4: Explain the difference in F1 scores between mistral_response and mistral_response_cleaned.**

# ### **Question 3: Few-Shot Prompting for Text Classification**

#

# ##### **Q3.1: Prepare examples for a few-shot prompt, formulate the prompt, and generate the Mistral response.**

# **Generate a set of gold examples by randomly selecting 10 instances of user_input and assistant_output from dataset ensuring a balanced representation with 2 examples from each class.**

# In[ ]:


# Define title of exercise
title = 'Few-Shot Text-to-Label Classification'


# In[ ]:




# In[ ]:


# Create training set data for few shot prompting.
examples_df = create_examples_df(data, PRODUCT_SAMPLE_SIZE)

# Create the training set by excluding examples.
gold_examples_df = data.drop(index=examples_df.index)

# Convert examples to JSON
columns_to_select = ['narrative', 'product']
examples_json = examples_df[columns_to_select].to_json(orient='records')

# Print the first record from the JSON
print('First record from JSON data: ' + str(json.loads(examples_json)[0]))

# Print the shapes of the datasets.
# Note: Gold examples (also called "golden examples" or "ground truth examples") are high-quality, pre-verified input-output pairs that serve as the "correct" or "ideal" examples for a specific task.
print("Examples Set Shape:", examples_df.shape)
print("Gold Examples Shape:", gold_examples_df.shape)

# - Define your **system_message**.
# - Define **first_turn_template**, **example_template** and **prediction template**
# - **create few shot prompt** using gold examples and system_message
# - Randomly select 30 rows from test_df as test_data
# - Create **mistral_response** with **mistral_response_cleaned** columns for this

# In[ ]:


# Few Shot Prompt System Message
system_message = f"""
System: You are an expert text classification model using few-shot prompting logic.
Your task is to classify a customer complaint into financial product categories: {labels_str}.
Use the examples provided to help you classify the new user input.
Only return the category name (that matches one of the product categories) and **nothing else**.
If the category name returns with a backslash in it, remove it!
"""

print(f'few-shot: system_message: {system_message}')

# In[ ]:


# Few Shot Templates for Mistral 7B

# ----- First Turn -----
first_turn_template = "<s>[INST]{system_message}\n\n{user_input}[/INST]{assistant_output}</s>"

# ----- Examples -----
examples_template = "<s>[INST]{user_input}[/INST]{assistant_output}</s>"

# ----- Predictions -----
prediction_template = "<s>[INST]{user_input}[/INST]"

# In[ ]:


# In[ ]:


# Get Mistral response for few shot prompt.
few_shot_prompt = create_few_shot_prompt(system_message, examples_df)
print(few_shot_prompt)

# In[ ]:




# In[ ]:


# Get data for a few shot prompt testing.
new_data = create_sample_data(TRAINING_DATA_SIZE, gold_examples_df)
narratives = new_data['narrative']

# In[ ]:


# Caution!!! This cell takes the longest to process (over 10 minutes in Google Colab)!
# Takes about 447.1453 seconds to run in Jupyter Notebooks.
# The function took 7m 17s 145ms seconds to run.
start_time = start_timer()
new_data['mistral_response'] = get_few_shot_mistral_response(narratives)
output_timer(start_time, title)

# In[ ]:


# Few Shot Mistral Response (cleaned).
start_time = start_timer()
new_data['mistral_response_cleaned'] = clean_mistral_response(new_data['mistral_response'])
output_timer(start_time, title)

# ##### **Q3.2: Calculate the F1 score**

# In[ ]:


# ----- Few Shot Output -----

# Calculate F1 score for 'product' and 'mistral_response'
product = new_data['product']
mistral_response = new_data['mistral_response']
mistral_response_cleaned = new_data['mistral_response_cleaned']

# Get F1 score to output
f1 = f1_score(product, mistral_response, average='micro')

show_banner(title)

# Few Shot Prompt for Text Classification Results
print(f'F1 Score: {f1}')

# Few-Shot Prompt for Text Classification Results
f1_cleaned = f1_score(product, mistral_response_cleaned, average='micro')

# Few Shot Prompt for Text Classification Results
print(f'Cleaned F1 Score: {f1_cleaned}')

# ##### **Q3.3: Share your observations on the few-shot and zero-shot prompt techniques.**

# # **Section 3: Text to Text generation**

# ### **Question 4: Zero-Shot Prompting for Text Summarization**

# ##### **Q4.1: Define the Prompt Template, System Message, generate prompt and model response**
#

# - Define a **system message** as a string and assign it to the variable system_message to generate summary of narrative in data.
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

# In[ ]:


title = 'Zero-Shot Text Summarization'

# In[ ]:


# Updated prompt message
system_message = """
You are an expert summarization tool for financial complaints. Your task is to provide a concise summary (1-3 sentences) of the complaint. The summary should focus on three key points:
1. The main problem or core issue.
2. The company or companies involved.
3. The customer's desired outcome or the current status of the problem.

If the complaint text is unclear or incomplete, leave the summary blank.
**Only provide the summary.**  Do not provide any additional text.
"""

print(f'system_message={system_message}')

# In[ ]:


# Zero-Shot prompting for Text Summarization.
zero_shot_prompt_template = "<s>[INST] {system_message}User Input: {user_input} [/INST]"

# ##### **Q4.2: Generate mistral_response column containing LLM generated summaries**

# In[ ]:


# Create test data with gold examples for zero shot text-to-text summarization.
gold_examples = create_sample_data(TEST_DATA_SIZE, data)
narratives = gold_examples['narrative']

# In[ ]:


# Caution: This cell takes a while to process!
# Takes about 83 seconds to run in Jupyter Notebook.
start_time = start_timer()
gold_examples['mistral_response'] = get_zero_shot_mistral_response(narratives)
output_timer(start_time, title)

# In[ ]:


# Confirm the new validation data has been loaded with the correct columns.
gold_examples.head()


# ##### **Q4.3: Evaluate bert score**

# In[ ]:


# In[ ]:

"""
def evaluate_score(test_data, scorer, bert_score=False):
    """
    Return the ROUGE score or BERTScore for predictions on gold examples
    For each example we make a prediction using the prompt.
    Gold summaries and the AI generated summaries are aggregated into lists.
    These lists are used by the corresponding scorers to compute metrics.
    Since BERTScore is computed for each candidate-reference pair, we take the
    average F1 score across the gold examples.

    Args:
        prompt (List): list of messages in the Open AI prompt format
        gold_examples (str): JSON string with list of gold examples
        scorer (function): Scorer function used to compute the ROUGE score or the
                           BERTScore
        bert_score (boolean): A flag variable that indicates if BERTScore should
                              be used as the metric.

    Output:
        score (float): BERTScore or ROUGE score computed by comparing model predictions
                       with ground truth
    """

    model_predictions = test_data['mistral_response'].tolist()
    ground_truths = test_data['summary'].tolist()

    if bert_score:
        score = scorer.compute(
            predictions=model_predictions,
            references=ground_truths,
            lang="en",
            rescale_with_baseline=True,
            model_type=BERT_SCORE_MODEL  # added!
        )

        return sum(score['f1']) / len(score['f1'])
    else:
        return scorer.compute(
            predictions=model_predictions,
            references=ground_truths,
            model_type=BERT_SCORE_MODEL  # added!
        )
"""

# In[ ]:


bert_scorer = evaluate.load("bertscore")

# In[ ]:


# Get the score for text-to-text summarization.  For this summarization we will use the BERT score.
start_time = start_timer()
score = evaluate_score(gold_examples, bert_scorer, True)
output_timer(start_time, title + ' BERT Score')

print(f'BERT Score: {score}')

# When evaluating Text-To-Text Summarization with the B.E.R.T Score we get a result of 0.320.
#
# However, if we add the parameter <code>model_type='distilbert-base-uncased'</code> it jumps to 0.515! The low score is because a zero-shot prompt returns a summary that differs from my gold-standard summaries in the data set. A score of 0.8 would be better.

#
#
# ---
# # End of Project
#


if __name__ == "__main__":

    # @link https://docs.python.org/3/library/argparse.html
    # Check for arguments to determine which processes to run.  Check for 'seed', 'eda' or no arguments to run the full pipeline.
    parser = argparse.ArgumentParser(
        prog='Financial Product Complaint Classification and Summarization',
        description='Analyzes financial customer complaints using Generative AI for text classification and summarization.',
        epilog='Example usage: python main.py --eda to run only the EDA pipeline.')

    args = parser.parse_args()
    print(args.filename, args.count, args.verbose)

    parser.add_argument('filename')           # positional argument
    parser.add_argument('-c', '--count')      # option that takes a value
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag


    # Run the EDA pipeline to understand the dataset and prepare it for modeling.
    #run_eda_pipeline()

    # Run the main pipeline to execute the text classification and summarization tasks.
    #run_main_pipeline()
# --- End Program --- #
