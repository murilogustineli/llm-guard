import os
import re
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd

# Hugging Face
from huggingface_hub import login
from transformers import pipeline

# Gaudi
# from habana_frameworks.torch.hpu import wrap_in_hpu_graph
# import habana_frameworks.torch.core as htcore


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the token file
path = os.path.join(script_dir, "token.txt")

with open(path, "r") as file:
    token = file.read().strip()

# Necessary for gated Hugging Face repos
login(token)


# Labels and Categories that will be used in the prompt
labels = ["false", "partially true", "mostly true", "true"]
category_type = {
    "health and medicine": [
        "COVID-19 (virus origins, treatments, lockdowns)",
        "vaccines (efficacy, safety, side effects)",
        "alternative medicine vs. scientific treatments",
        "diet fads and health benefits",
        "drug claims and cures",
    ],
    "politics and government": [
        "election integrity",
        "politicians or political events",
        "conspiracy theories around governmental institutions",
        "political narratives",
        "claims about political policies or actions",
    ],
    "climate change and environmental issues": [
        "facts about climate change",
        "renewable energy",
        "environmental disasters (e.g., wildfires, floods)",
        "conservation efforts or endangered species",
    ],
    "science and technology": [
        "flat Earth theories or other pseudoscientific claims",
        "space exploration",
        "AI fears",
        "5G technology",
    ],
    "conspiracy theories": [
        "QAnon or other large-scale conspiracy movements",
        "9/11 and false flag theories",
        "anti-science movements (e.g., anti-vaccine, anti-climate change)",
        "secret societies controlling world governments",
    ],
    "economics and financial markets": [
        "market crashes or investment schemes",
        "pyramid schemes and financial advice",
        "cryptocurrencies or blockchain",
        "economic collapse or manipulation",
    ],
    "social and cultural issues": [
        "racial, ethnic, or religious groups",
        "gender identity or LGBTQ+",
        "bias or selective reporting in media",
        "historical events",
    ],
    "technology and AI": [
        "AI or machine learning capabilities",
        "deepfake videos and synthetic content spread",
        "tech privacy breaches or hacking",
    ],
}


prompt = """
    I am generating synthetic OUTPUT to fine-tune
    my BERT model for detecting misinformation. The goal is to classify
    text based on its accuracy and prevent the spread of false information.

    Label definitions:
    - false: Completely untrue or fabricated information.
    - partially true: Contains some truth but is misleading or lacks important context.
    - mostly true: Largely accurate but may have minor inaccuracies or omissions.
    - true: Entirely accurate and factual information.

    Follow these steps to generate an accurate OUTPUT:

    To ensure the generated OUTPUT matches the LABEL, follow this step-by-step reasoning:
    1. Consider the CATEGORY and TYPE to understand the topic.
    2. Think about common statements or beliefs that align with the LABEL.
    3. Reflect on whether the statement should be false, partially true, mostly true, or true.
    4. Generate an OUTPUT that aligns with the assigned LABEL and category.

    Examples:

    LABEL: false
    CATEGORY: health and medicine
    TYPE: vaccines (efficacy, safety, side effects)
    OUTPUT: Vaccines contain microchips to track people.
    REASONING: This statement is false, because vaccines are developed to prevent diseases by triggering an immune response, not to track individuals. The ingredients in vaccines are publicly disclosed by manufacturers and are subject to rigorous testing and regulation by health authorities like FDA and WHO.

    LABEL: mostly true
    CATEGORY: climate change and environmental issues
    TYPE: conservation efforts or endangered species
    OUTPUT: Polar bear populations are declining due to melting Arctic ice.
    REASONING: This statement is mostly true because many polar bear populations are under threat as their sea ice habitat diminishes due to climate change. However, it may not be universally true for all polar bear populations, as some subpopulations have remained stable or even increased. The statement is largely accurate but may lack nuance regarding regional differences.

    LABEL: partially true
    CATEGORY: health and medicine
    TYPE: diet fads and health benefits
    OUTPUT: Eating carrots significantly improves your night vision.
    REASONING: This contains some truth because carrots are rich in beta-carotene, which is good for overall eye health, but the idea that they significantly improve night vision is misleading. The myth originated from British propaganda during World War II to explain their pilots' improved night-time operations.

    LABEL: true
    CATEGORY: health and medicine
    TYPE: COVID-19 (virus origins, treatments, lockdowns)
    OUTPUT: Wearing masks can help reduce the transmission of respiratory viruses.
    REASONING: This is true and factual information. Numerous studies have shown that masks are effective in reducing the spread of viruses, including the novel coronavirus.
    ######################################
    Your task is to generate one OUTPUT for the classification below. It's extremely important that the generated OUTPUT aligns with the assigned LABEL.
    Only return the OUTPUT and REASONING. Do not return the LABEL, CATEGORY, or TYPE.
    """


def extract_quoted_text(text):
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


def diversify(category):
    """
    Randomly selects a value from the list associated with a given key in the category_type dictionary.

    Args:
        category (str): A key in the category_type dictionary.

    Returns:
        str: A randomly chosen value from the list associated with the provided key.
    """
    return np.random.choice(category_type[category])


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
    categories: str,
    prompt: str,
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
    print(f"Running on {device}.")

    # Generate filename with current date, time, and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.split("/")[-1]

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{timestamp}_{model_name}.csv")

    # If sample_size is not divisible by batch_size, an extra batch is added
    num_batches = (sample_size + batch_size - 1) // batch_size

    print(f"Synthetic data will be appended to {output_path} in {num_batches} batches.")

    for batch in range(num_batches):
        # Calculate the start and end indices for the current batch
        start = batch * batch_size
        end = min(start + batch_size, sample_size)

        # Store results of the current batch
        batch_data = []

        # Assign random labels to the current batch
        batch_random_labels = np.random.choice(labels, batch_size, replace=True)

        # Assign random categories to the current batch
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

            # Get results from Llama
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant designed to generate synthetic data with labels {labels} in categories {list(category_type.keys())}.",
                },
                {"role": "user", "content": prompt_input},
            ]
            generator = pipeline("text-generation", model=model, device=device)
            result = generator(messages, max_new_tokens=max_new_tokens)[0][
                "generated_text"
            ][-1]["content"]

            # Uncomment to see the raw outputs
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

        # Convert the batch results to a DataFrame
        batch_df = pd.DataFrame(batch_data)

        # Append the DataFrame to the CSV file
        if batch == 0:
            # If it's the first batch, write headers
            batch_df.to_csv(output_path, mode="w", index=False)
        else:
            # For subsequent batches, append without headers
            batch_df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"Saved batch number {batch + 1}/{num_batches}")


def parse_args():
    parser = ArgumentParser()
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


if __name__ == "__main__":
    args = parse_args()

    # generate synthetic data
    sdg(
        sample_size=args.sample_size,
        labels=labels,
        categories=list(category_type.keys()),
        prompt=prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_hpu=args.use_hpu,
        output_dir="./",
        model=args.model,
        verbose=args.verbose,
    )

# MODELS
# meta-llama/Meta-Llama-3.1-8B-Instruct
# mistralai/Mixtral-8x7B-Instruct-v0.1
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
