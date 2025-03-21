import os
import re
from argparse import ArgumentParser
from datetime import datetime
from prompts import PROMPT_V1, PROMPT_V2
from labels_categories import LABELS, CATEGORY_TYPE
from config import load_hf_token

import numpy as np
import pandas as pd

# Hugging Face
from transformers import pipeline

# Gaudi
# from habana_frameworks.torch.hpu import wrap_in_hpu_graph
# import habana_frameworks.torch.core as htcore


def extract_quoted_text(text: str) -> str:
    """
    Extracts the longest quoted text from the LLM's output.

    Args:
        text (str): The generated text from the Assistant.

    Returns:
        str: The longest quoted text found within the input string.
    """
    # Regex pattern to match quoted strings
    quotes = re.findall(r'"(.*?)"', text)

    # Find the longest quote
    if quotes:
        longest_quote = max(quotes, key=len)
        return longest_quote
    else:
        return text


def diversify(category: str):
    """
    Randomly selects a value from the list associated with a given key in the category_type dictionary.

    Args:
        category (str): A key in the category_type dictionary.

    Returns:
        str: A randomly chosen value from the list associated with the provided key.
    """
    return np.random.choice(CATEGORY_TYPE[category])


def extract_output_and_reasoning(result: str) -> tuple:
    # Use re.DOTALL to make '.' match newline characters
    match = re.search(r"OUTPUT:\s*(.*?)\s*REASONING:\s*(.*)", result, re.DOTALL)
    if match:
        output_text = match.group(1).replace("\n", " ").strip()
        reasoning_text = match.group(2).replace("\n", " ").strip()
        return output_text, reasoning_text
    else:
        return None, None


def sdg(
    sample_size: int,
    labels: str,
    category_type: dict,
    prompt_version: str,
    batch_size: int = 20,
    max_new_tokens: int = 250,
    use_hpu: bool = True,
    output_dir: str = "./data",
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    verbose: bool = False,
):
    """
    Generates synthetic data based on specified categories and labels.

    Args:
        sample_size (int): The number of synthetic data samples to generate.
        labels (list of str): The labels used to classify the synthetic data.
        categories (list of str): The categories for data generation and diversification.
        batch_size (int): The number of samples per batch to append to the output file.
        use_hpu (bool): If True, the function will use Gaudi hardware; otherwise, it will use CPU.
        output_dir (str): The directory path where the output file will be saved.
        model (str): The large language model used for generating the synthetic data.
    """

    device = "hpu" if use_hpu else "cpu"
    # tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"Running on {device}")

    # Generate filename with current date, time, and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.split("/")[-1]

    # set prompt for synthetic data generation
    if prompt_version == "v1":
        prompt = PROMPT_V1
    elif prompt_version == "v2":
        prompt = PROMPT_V2
    else:
        raise ValueError("Prompt must be either 'v1' or v2'.")
    print(f"Using prompt version: {prompt_version}")
    print(f"Prompt: {prompt}")

    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # output path to write synthetic data to a CSV file
    output_path = os.path.join(
        output_dir, f"{timestamp}_{model_name}_prompt-{prompt_version}.csv"
    )

    # if sample_size is not divisible by batch_size, an extra batch is added
    num_batches = (sample_size + batch_size - 1) // batch_size

    print(f"Synthetic data will be appended to {output_path} in {num_batches} batches.")

    for batch in range(num_batches):
        # calculate the start and end indices for the current batch
        start = batch * batch_size
        end = min(start + batch_size, sample_size)

        # store results of the current batch
        batch_data = []

        # assign random labels to the current batch
        batch_random_labels = np.random.choice(labels, batch_size, replace=True)

        # assign random categories to the current batch
        categories = list(category_type.keys())
        batch_random_categories = np.random.choice(categories, batch_size, replace=True)

        for i in range(start, end):
            prompt_input = (
                prompt
                + f"""
                LABEL: {batch_random_labels[i - start]}
                CATEGORY: {batch_random_categories[i - start]}
                TYPE: {diversify(batch_random_categories[i - start])}
                OUTPUT:
                REASONING:
                """
            )

            # generate results from the model
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant designed to generate synthetic data with labels {labels} in categories {categories}.",
                },
                {"role": "user", "content": prompt_input},
            ]
            generator = pipeline("text-generation", model=model, device=device)
            result = generator(messages, max_new_tokens=max_new_tokens)[0][
                "generated_text"
            ][-1]["content"]

            # extract the output and reasoning from the result
            output, reasoning = extract_output_and_reasoning(result)
            if verbose:
                print(f"OUTPUT: {output}")
                print(f"REASONING: {reasoning}")

            batch_data.append(
                {
                    "output": output,
                    "reasoning": reasoning,
                    "label": batch_random_labels[i - start],
                    "model": model,
                }
            )

        # convert the batch results to a DataFrame
        batch_df = pd.DataFrame(batch_data)

        # append the DataFrame to the CSV file
        if batch == 0:
            # If it's the first batch, write headers
            batch_df.to_csv(output_path, mode="w", index=False)
        else:
            # for subsequent batches, append without headers
            batch_df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"Saved batch number {batch + 1}/{num_batches}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1",
        help="Prompt version to be used to generate synthetic data (v1 or v2).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples generated by the LLM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Size of the batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=250,
        help="Max number of tokens generated in the output.",
    )
    parser.add_argument(
        "--use-hpu",
        type=bool,
        default=False,
        help="Flag to use HPU. If False, use CPU",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model used to generate synthetic data.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Prints the OUTPUT and REASONING being generated by the model.",
    )
    return parser.parse_args()


def get_output_dir():
    # absolute path to the repository root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    output_dir = os.path.join(repo_root, "data")
    return output_dir


if __name__ == "__main__":
    # load Hugging Face token to access models
    load_hf_token()

    # parse command-line arguments
    args = parse_args()

    # generate synthetic data
    sdg(
        sample_size=args.sample_size,
        labels=LABELS,
        category_type=CATEGORY_TYPE,
        prompt_version=args.prompt_version,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_hpu=args.use_hpu,
        output_dir=get_output_dir(),
        model=args.model,
        verbose=args.verbose,
    )

# MODELS TO CHOOSE FROM:
# meta-llama/Meta-Llama-3.1-8B-Instruct
# mistralai/Mixtral-8x7B-Instruct-v0.1
