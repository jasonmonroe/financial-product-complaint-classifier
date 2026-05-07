# src/modeling.py
try:
    from IPython.display import display
except ImportError:
    display = print

from sklearn.metrics import f1_score
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import pandas as pd
import evaluate

from src.config import MODEL_PATH, MODEL_BASENAME, MODEL_ATTRS, MISTRAL_ATTRS, BERT_SCORE_MODEL
from src.utils import start_timer, output_timer, show_banner, extract_category, create_sample_data

def init_model():
    # Download the model from Hugging Face Hub and get the local path.
    return hf_hub_download(repo_id=MODEL_PATH, filename=MODEL_BASENAME)

def llama() -> Llama:
    """Initializes and returns the Llama model instance with configured attributes."""
    return Llama(
        model_path=init_model(),
        n_threads=MODEL_ATTRS['cpu_cores'],     # CPU cores
        n_batch=MODEL_ATTRS['batch_size'],      # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=MODEL_ATTRS['gpu_layers'], # Change this value based on your model and your GPU VRAM pool.
        n_ctx=MODEL_ATTRS['context_window'],    # Context window
    )

# Format zero shot prompt.
def format_zero_shot_prompt(system_message: str, user_input: str, zero_shot_prompt_template: str) -> str:
    prompt = zero_shot_prompt_template.format(
        system_message=system_message,
        user_input=user_input
    )

    return prompt

# Generate prompt response with Mistral.
def generate_zero_shot_mistral_response(llm: Llama, system_message: str, input_text: str, template: str) -> str:
    prompt = format_zero_shot_prompt(system_message, input_text, template)
    return generate_prompt_response(llm, prompt)

# Apply the mistral response function on every row value in the "narrative" column.
def get_zero_shot_mistral_response(llm, narratives, system_message: str, template: str):
    return narratives.apply(lambda x: generate_zero_shot_mistral_response(llm, system_message, x, template))


def create_few_shot_prompt(system_message, examples_df, first_turn_template, examples_template):

    """
    Return a prompt message in the format expected by Mistral 7b.
    10 examples are selected randomly as golden examples to form the
    few-shot prompt.
    We then loop through each example and parse the narrative as the user message
    and the product as the assistant message.

    Args:
        system_message (str): system message with instructions for classification
        examples (DataFrame): A DataFrame with examples (product + narrative + summary)
        to form the few-shot prompt.

    Output:
        few_shot_prompt (str): A prompt string in the Mistral format
    """

    few_shot_prompt = ''

    # Convert to dictionary directly for better performance
    for idx, example in enumerate(examples_df[['narrative', 'product']].to_dict(orient='records')):
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
def format_few_shot_prompt(few_shot_prompt: str, new_review: str, prediction_template: str) -> str:
    return few_shot_prompt + prediction_template.format(user_input=new_review)

# Generate prompt response with Mistral
def generate_few_shot_mistral_response(llm: Llama, few_shot_prompt: str, input_text: str, prediction_template: str) -> str:
    prompt = format_few_shot_prompt(few_shot_prompt, input_text, prediction_template)
    return generate_prompt_response(llm, prompt)

# Gets Mistral response for few shot prompts.
def get_few_shot_mistral_response(llm, narratives, few_shot_context: str, prediction_template: str):
    return narratives.apply(lambda x: generate_few_shot_mistral_response(llm, few_shot_context, x, prediction_template))

# Generate response from prompt.  This will handle zero and few shot responses.
# Mistral model extends from Llama (model).
def generate_prompt_response(llm: Llama, prompt: str) -> str:
    response = llm(
        prompt=prompt,
        max_tokens=MISTRAL_ATTRS['max_tokens'],
        temperature=MISTRAL_ATTRS['temperature'],
        top_p=MISTRAL_ATTRS['top_p'],
        repeat_penalty=MISTRAL_ATTRS['repeat_penalty'],
        top_k=MISTRAL_ATTRS['top_k'],
        stop=MISTRAL_ATTRS['stop_sequences'],
        echo=MISTRAL_ATTRS['echo']
    )

    return response["choices"][0]["text"]

def clean_mistral_response(mistral_responses):
    """Applies category extraction and string cleaning to model outputs."""
    return mistral_responses.apply(lambda x: extract_category(x.replace('\\', '').strip()))

def evaluate_score(test_data, scorer, bert_score=False):
    """Calculates BERTScore or ROUGE score for model predictions."""
    model_predictions = test_data['mistral_response'].tolist()
    ground_truths = test_data['summary'].tolist()

    if bert_score:
        score = scorer.compute(
            predictions=model_predictions,
            references=ground_truths,
            lang="en",
            rescale_with_baseline=True,
            model_type=BERT_SCORE_MODEL
        )
        return sum(score['f1']) / len(score['f1'])

    return scorer.compute(
        predictions=model_predictions,
        references=ground_truths,
        model_type=BERT_SCORE_MODEL
    )
