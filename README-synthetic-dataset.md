---
task_categories:
- text-classification
languages:
- en
license: TBD (apache-2.0 ? GPL)
---

# Synthetic Text Classification Dataset

- **Dataset type**: Synthetic  
- **Number of samples**: 50,000  
- **Task**: Text Classification  
- **Domain**: Multi-label classification of text into polite, somewhat polite, neutral, and impolite categories

## Dataset Description
This dataset was generated to train and evaluate models on the task of text classification according to politeness. Synthetic data generation was carried out by a custom designed pipeline using the following LLMs:

- [Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b-it)
- [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

### Structure
The dataset contains the following splits:

- **train**: 45,000 examples
- **validation**: 5,000 examples

Each example contains:

- **text**: The text input (string)
- **label**: The classification label (category: polite, somewhat polite, neutral, and impolite)
- **model**: LLM name used to generate text

## Description of labels
- **polite**: Text is considerate and shows respect and good manners, often including courteous phrases and a friendly tone.
- **somewhat polite**: Text is generally respectful but lacks warmth or formality, communicating with a decent level of courtesy.
- **neutral**: Text is straightforward and factual, without emotional undertones or specific attempts at politeness.
- **impolite**: Text is disrespectful or rude, often blunt or dismissive, showing a lack of consideration for the recipient's feelings.

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("Intel/polite-guard-synthetic")
```
## Disclaimer
Polite Guard has been trained and validated on a limited set of data
that pertains to customer reviews, product reviews, and corporate
communications. Accuracy metrics cannot be guaranteed outside these
narrow use cases, and therefore this tool should be validated within
the specific context of use for which it might be deployed. This tool
is not intended to be used to evaluate employee performance. This tool
is not sufficient to prevent harm in many contexts, and additional
tools and techniques should be employed in any sensitive use case
where impolite speech may cause harm to individuals, communities, or
society.