import os
import re
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
labels = ["polite", "somewhat polite", "neutral", "impolite"]
category_type = {
    "travel": [
        "business",
        "luxury",
        "budget or economy",
        "cultural",
        "medical",
        "air",
        "train",
        "cruises and ferries",
        "bus or car rental",
    ],
    "food and drink": [
        "pizza",
        "international",
        "regional",
        "fusion",
        "dessert",
        "vegetarian",
        "halal",
        "bakery",
        "street food",
        "buffet",
        "fast food",
        "local and organic",
        "coffee",
        "bar",
        "gluten-free",
    ],
    "stores": [
        "apparel and accessories",
        "electronics and appliances",
        "grocery and food",
        "health and beauty",
        "home and furniture",
        "sports and outdoors",
        "toys and games",
    ],
    "finance": ["banking", "credit", "insurance", "loans", "fees and charges"],
    "professional development": [
        "technical skills",
        "soft skills",
        "creative skills",
        "workshop",
        "bootcamp",
        "integration training",
    ],
    "sports clubs": [
        "team sports",
        "individual sports",
        "racket sports",
        "water sports",
        "winter sports",
        "combat sports",
    ],
    "cultural and educational": [
        "museum",
        "theater",
        "zoo or aquarium",
        "art gallery",
        "botanical garden",
        "library",
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
            prompt = f"""I am creating synthetic OUTPUT to fine-tune
            my BERT model. The use-case is customer service chatbots.
            You should generate only one OUTPUT for the classification
            LABEL: {batch_random_labels[i - start]} in CATEGORY:
            {batch_random_categories[i - start]} and TYPE
            {diversify(batch_random_categories[i - start])}. Feel free
            to diversify the output further according to the age and
            seasonality when applicable. Avoid phrases such as "The
            issue with", "Are you kidding me", and "Hey, I" as much as possible.

            Examples.
            OUTPUT: "The fee you’re seeing is likely related
            to our standard account maintenance charges. I can provide
            more details if needed."

            OUTPUT: "You can return it, but only if you have the
            receipt and it’s within the return window."

            OUTPUT: "It's not our fault your baggage didn't make it.
            What do you expect us to do about it now?"

            OUTPUT: "I apologize for the trouble you’ve had with the
            heater. We can certainly look into a return or exchange.
            Please bring in your receipt, and we’ll take care of it
            for you."

            Only return one OUTPUT and not the LABEL or the CATEGORY.
            """
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant designed to generate synthetic customer service data with labels {labels} in categories {list(category_type.keys())}.",
                },
                {"role": "user", "content": prompt},
            ]
            generator = pipeline("text-generation", model=model, device=device)
            result = generator(messages, max_new_tokens=128)[0]["generated_text"][-1][
                "content"
            ]

            # Uncomment to see the raw outputs
            # print(result)

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


sdg(
    sample_size=25,
    labels=labels,
    categories=list(category_type.keys()),
    batch_size=25,
    use_hpu=False,
    output_dir="./",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
)
