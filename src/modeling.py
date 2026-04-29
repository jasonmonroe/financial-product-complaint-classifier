# src/modeling.py

from src.config import *

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.utils import extract_category


def init_model():
    # Download the model from Hugging Face Hub and get the local path.
    return hf_hub_download(repo_id=MODEL_PATH, filename=MODEL_BASENAME)

def llama():
    # Load and create an instance of the Llama c++ model.
    lcpp_llm = Llama(
        MODEL_PATH=init_model(),
        n_threads=MODEL_ATTRS['cpu_cores'],     # CPU cores
        n_batch=MODEL_ATTRS['batch_size'],      # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=MODEL_ATTRS['gpu_layers'], # Change this value based on your model and your GPU VRAM pool.
        n_ctx=MODEL_ATTRS['context_window'],    # Context window
    )

    return lcpp_llm

def generate_prompt(system_message,user_input):
    prompt=zero_shot_prompt_template.format(system_message=system_message,user_message=user_input)
    return prompt


# Format zero shot prompt.
def format_zero_shot_prompt(system_message: str, user_input: str) -> str:
    prompt = zero_shot_prompt_template.format(
        system_message=system_message,
        user_input=user_input
    )

    return prompt


# Generate prompt response with Mistral.
def generate_zero_shot_mistral_response(input_text: str) ->str:
    prompt = format_zero_shot_prompt(system_message, input_text)
    return generate_prompt_response(prompt)


# Apply the mistral response function on every row value in the "narrative" column.
def get_zero_shot_mistral_response(narratives):
    return narratives.apply(lambda x: generate_zero_shot_mistral_response(x))



def create_few_shot_prompt(system_message, examples_df):

    """
    Return a prompt message in the format expected by Mistral 7b.
    10 examples are selected randomly as golden examples to form the
    few-shot prompt.
    We then loop through each example and parse the narrative as the user message
    and the product as the assistant message.

    Args:
        system_message (str): system message with instructions for classification
        examples(DataFrame): A DataFrame with examples (product + narrative + summary)
        to form the few-shot prompt.

    Output:
        few_shot_prompt (str): A prompt string in the Mistral format
    """

    few_shot_prompt = ''

    columns_to_select = ['narrative', 'product']
    examples = (
        examples_df.loc[:, columns_to_select].to_json(orient='records')
    )

    for idx, example in enumerate(json.loads(examples)):
        user_input_example = example['narrative']
        assistant_output_example = example['product']

        if idx == 0:
            few_shot_prompt += first_turn_template.format(
                system_message=system_message,
                user_input=user_input_example,
                assistant_output=assistant_output_example
            )
        else:
            few_shot_prompt += examples_template.format(
                user_input=user_input_example,
                assistant_output=assistant_output_example
            )

    return few_shot_prompt


# Pass few_shot_prompt that was created and the review examples which are actually the narratives.
def format_few_shot_prompt(few_shot_prompt: str, new_review: str) -> str:
    prompt = few_shot_prompt + prediction_template.format(user_input=new_review)
    return prompt


# Generate prompt response with Mistral
def generate_few_shot_mistral_response(input_text: str) -> str:
    prompt = format_few_shot_prompt(few_shot_prompt, input_text)
    return generate_prompt_response(prompt)


# Gets Mistral response for few shot prompts.
def get_few_shot_mistral_response(narratives):
    return narratives.apply(lambda x: generate_few_shot_mistral_response(x))

# Generate response from prompt.  This will handle zero and few shot responses.
# Mistral model extends from Llama (model).
def generate_prompt_response(prompt: str) -> str:
    response = lcpp_llm(
        prompt=prompt,
        max_tokens=MISTRAL_ATTRS['max_tokens'],
        temperature=MISTRAL_ATTRS['temperature'],
        top_p=MISTRAL_ATTRS['top_p'],
        repeat_penalty=MISTRAL_ATTRS['repeat_penalty'],
        top_k=MISTRAL_ATTRS['top_k'],
        stop=MISTRAL_ATTRS['stop_sequences'],
        echo=MISTRAL_ATTRS['echo']
    )
    response_text = response["choices"][0]["text"]
    print(response_text)
    return response_text


# Clean up the mistral response by extracting the category.
# Strip backslash if found from the product category.
def clean_mistral_response(mistral_responses):
    # return mistral_responses.apply(lambda x: extract_category(x))
    return mistral_responses.apply(lambda x: extract_category(x.replace('\\', '').strip()))


# --- Run Prompts --- #
def run_zero_shot_prompt():
    pass

def run_few_shot_prompt():
    pass


def run_few_shot_text_classification():
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

    labels, label_pattern, labels_str = get_unique_product_labels(df)

    # Few Shot Prompt System Message
    system_message = f"""
        System: You are an expert text classification model using few-shot prompting logic.
        Your task is to classify a customer complaint into financial product categories: {labels_str}.
        Use the examples provided to help you classify the new user input.
        Only return the category name (that matches one of the product categories) and **nothing else**.
        If the category name returns with a backslash in it, remove it!
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

    # Few Shot Mistral Response (cleaned).
    start_time = start_timer()
    df_sample['mistral_response_cleaned'] = clean_mistral_response(df_sample['mistral_response'])
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


def run_zero_shot_text_summarization():
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
    narratives = df_gold_examples['narrative']

    start_time = start_timer()
    df_gold_examples['mistral_response'] = get_zero_shot_mistral_response(narratives)
    output_timer(start_time, title)

    bert_scorer = evaluate.load('bertscore')

    # Get the score for text-to-text summarization.  For this summarization we will use the BERT score.
    start_time = start_timer()
    score = evaluate_score(df_gold_examples, bert_scorer, True)
    output_timer(start_time, title + ' BERT Score')

    print(f'BERT Score: {score}')

def run_zero_shot_text_classification():
    title = 'Zero-Shot Prompting for Text Classification'

    label_pattern, labels_str = get_labels()

    system_message = f"""
    System: You are an expert text classification model.
    Your task is to classify a customer complaint into one of the following product categories: {labels_str}.
    **Only return the category name** and nothing else.
    If the product category name has a backslash in it, remove it.
    """

    # Define the template used for prompting the labels.
    zero_shot_prompt_template = """
    <s>[INST] {system_message}
    
    User Input: {user_input}
    Category: [/INST]
    """





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


    pass

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


def get_bert_score():
    pass


