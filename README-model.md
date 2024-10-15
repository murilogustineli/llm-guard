---
language: en
license: TBD (apache-2.0 ? GPL)
datasets:
- Intel/polite-guard
tags:
- bert
- synthetic-data
- fine-tuning
- text-classification
---

# Fine-Tuned BERT Model for Politeness Classification

**Model type**: BERT (Bidirectional Encoder Representations from Transformers)  
**Architecture**: Fine-tuned BERT-base  
**Task**: Text Classification

## Model Description

This model is a fine-tuned version of [BERT-base](https://huggingface.co/bert-base-uncased) trained on a synthetic dataset generated to tackle classification of chatbot responses as polite, somewhat polite, neutral, or impolite.

## Datasets Used
This model was fine-tuned using the [Synthetic Text Classification Dataset](https://huggingface.co/datasets/Intel/polite-guard-synthetic) and evaluated on the [Synthetic Benchmark Dataset](https://huggingface.co/datasets/Intel/polite-guard-benchmark).

### Key Features:
- Fine-tuned on a synthetic dataset created for politeness multi-label classification.
- Designed to perform well on text related to finance, travel, food, sports, education, and business.

### Use Cases:
- Ensuring polite interactions with customer service or automated chatbot systems
- Protection against adversarial attacks. 

## Description of labels
- **polite**: Text is considerate and shows respect and good manners, often including courteous phrases and a friendly tone.
- **somewhat polite**: Text is generally respectful but lacks warmth or formality, communicating with a decent level of courtesy.
- **neutral**: Text is straightforward and factual, without emotional undertones or specific attempts at politeness.
- **impolite**: Text is disrespectful or rude, often blunt or dismissive, showing a lack of consideration for the recipient's feelings.
  
## Model Details

- **Training Data**: This model was trained on a synthetic dataset generated specifically for multi-label classification of text according to politeness. 
- **Base Model**: The base model is [BERT-base](https://huggingface.co/bert-base-uncased), with 12 layers, 110M parameters.
- **Fine-tuning Process**: Fine-tuning was performed on polite-guard synthetic dateset for 5 epochs, using  learning rate 2e-05, with AdamW optimizer that is the default optimizer in Hugging Face `Trainer` framework with weight decay. The best hyperparameters for fine-tuning were selected by a grid search on the following grid:

```python
param_grid = {
    'learning_rate': [5e-5, 3e-5, 2e-5],
    'per_device_train_batch_size': [16, 32],
    'num_train_epochs': [3, 4, 5],
}
```
Early stopping and model checkpointing were used in fine-tuning BERT to ensure optimal training. 

### Performance

Here are the key performance metrics for both the synthetic data and the manually annotated benchmark dataset:

- **Accuracy**: 87% on the polite-guard synthetic dataset.
- **F1-Score**: 87% on the polite-guard synthetic dataset.

- **Accuracy**: tbm% on the polite-guard benchmark dataset.
- **F1-Score**: tbm% on the polite-guard benchmark dataset.

---

## How to Use

This fine-tuned BERT model can be used for categorizing text into classes polite, somewhat polite, neutral, and impolite.

### Load the model and tokenizer:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Intel/polite-guard")
model = AutoModelForSequenceClassification.from_pretrained("Intel/polite-guard")

# Example usage
inputs = tokenizer("Your input text", return_tensors="pt")
outputs = model(**inputs)
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

## Citation
If you use this model, please cite:
@misc{Intel_polite-guard_2024,
  author = {Ehssan K https://github.com/ekintel},
  title = {Polite Guard},
  year = {2024},
  url = {https://huggingface.co/Intel/polite-guard},
}

