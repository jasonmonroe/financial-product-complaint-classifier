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


def run_zero_shot_prompt():
    pass

def run_few_shot_prompt():
    pass


def run_zero_shot_text_summarization():
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