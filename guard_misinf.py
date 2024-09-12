import os
import re
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd

# Hugging Face
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        "diet fads and misleading health benefits",
        "misleading drug claims and fake cures",
    ],
    "politics and government": [
        "election integrity (fraud claims, vote manipulation)",
        "misinformation about politicians or political events",
        "conspiracy theories around governmental institutions",
        "propaganda or distorted political narratives",
        "false claims about political policies or actions",
    ],
    "climate change and environmental issues": [
        "denial or misleading facts about climate change",
        "false claims about renewable energy",
        "myths around environmental disasters (e.g., wildfires, floods)",
        "misinformation about conservation efforts or endangered species",
    ],
    "science and technology": [
        "flat Earth theories or other pseudoscientific claims",
        "misleading information about space exploration (e.g., moon landing hoaxes)",
        "AI fears and misconceptions (e.g., AI taking over the world)",
        "5G technology myths (health risks, surveillance)",
    ],
    "conspiracy theories": [
        "QAnon or other large-scale conspiracy movements",
        "9/11 and false flag theories",
        "anti-science movements (e.g., anti-vaccine, anti-climate change)",
        "secret societies controlling world governments",
    ],
    "economics and financial markets": [
        "false claims about market crashes or investment schemes",
        "pyramid schemes and misleading financial advice",
        "misleading information about cryptocurrencies or blockchain",
        "unsubstantiated claims about economic collapse or manipulation",
    ],
    "social and cultural issues": [
        "misinformation surrounding racial, ethnic, or religious groups",
        "gender identity or LGBTQ+ related misinformation",
        "extreme bias or selective reporting in media",
        "misinformation about significant historical events",
    ],
    "technology and AI": [
        "myths about AI or machine learning capabilities (AI will replace all jobs)",
        "deepfake videos and synthetic content spread",
        "misleading information on tech privacy breaches or hacking",
    ],
}


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


def sdg(
    sample_size,
    labels,
    categories,
    batch_size=20,
    use_hpu=True,
    output_dir="./",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
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

    # Generate filename with current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

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
            prompt = f"""I am generating synthetic OUTPUT to fine-tune
            my BERT model for detecting misinformation. The goal is to classify
            text based on its accuracy and prevent the spread of false information.
            Your task is to generate one OUTPUT for the classification:

            LABEL: {batch_random_labels[i - start]}
            CATEGORY: {batch_random_categories[i - start]}
            TYPE: {diversify(batch_random_categories[i - start])}

            Label definitions:
            - False: Completely untrue or fabricated information.
            - Partially True: Contains some truth but is misleading or lacks important context.
            - Mostly True: Largely accurate but may have minor inaccuracies or omissions.
            - True: Entirely accurate and factual information.

            Follow these steps to generate an accurate OUTPUT:

            To ensure the generated OUTPUT matches the LABEL, follow this step-by-step reasoning:
            1. Consider the CATEGORY and TYPE to understand the topic.
            2. Think about common statements or beliefs that align with the LABEL.
            3. Reflect on whether the statement should be false, partially true, mostly true, or true.
            4. Generate an OUTPUT that aligns with the assigned LABEL and category.

            ### Chain-of-Thought Examples:

            - Reasoning: People often spread conspiracy theories about vaccines.
            - OUTPUT: "Vaccines contain microchips to track people."

            - Reasoning: Scientific studies confirm the effectiveness of vaccines.
            - OUTPUT: "Vaccines help the body develop immunity without causing the disease."

            - Reasoning: Some concerns about mail-in voting have been raised, but most are unfounded.
            - OUTPUT: "While mail-in voting can have occasional issues, it's generally safe and secure for the vast majority of voters."

            - Reasoning: Climate change denial is common among certain groups.
            - OUTPUT: "Climate change is a hoax created to control the economy."

            - Reasoning: While scientific consensus supports human-driven climate change, some debates exist about the timeline of impacts.
            - OUTPUT: "The majority of climate scientists agree that human activity is causing global warming, but there are still discussions about the exact rate of change."

            It's extremely important that the generated OUTPUT aligns with the assigned LABEL.
            Only return the OUTPUT and do not return the LABEL, the CATEGORY, or the reasoning for the output.
            """

            # Get results from Llama
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant designed to generate synthetic data with labels {labels} in categories {list(category_type.keys())}.",
                },
                {"role": "user", "content": prompt},
            ]
            generator = pipeline("text-generation", model=model, device=device)
            result = generator(messages, max_new_tokens=128)[0]["generated_text"][-1][
                "content"
            ]

            # Uncomment to see the raw outputs
            print(result)

            result = extract_quoted_text(result)
            batch_data.append(
                {
                    "text": result,
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
        default=25,
        help="Number of samples generated by the LLM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Size of the batch.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # generate synthetic data
    sdg(
        sample_size=args.sample_size,
        labels=labels,
        categories=list(category_type.keys()),
        batch_size=args.batch_size,
        use_hpu=args.use_hpu,
        output_dir="./",
        model=args.model,
    )
