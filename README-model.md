---
language: en
license: TBD (apache-2.0 ? GPL)
datasets:
- Intel/misinformation-guard
tags:
- bert
- synthetic-data
- fine-tuning
- text-classification
---


# Fine-Tuned BERT Model for Misinformation Classification
**Model type**: BERT (Bidirectional Encoder Representations from Transformers)  
**Architecture**: Fine-tuned BERT-base  
**Task**: Text Classification


## Model Description
This model is a fine-tuned version of [BERT-base](https://huggingface.co/bert-base-uncased) trained on a synthetic dataset generated to tackle classification of chatbot responses as `false`, `partially true`, `mostly true`, and `true` categories.


## Datasets Used
This model was fine-tuned using the [Synthetic Text Classification Dataset](https://huggingface.co/datasets/Intel/misinformation-guard-synthetic) and evaluated on the [Synthetic Benchmark Dataset](https://huggingface.co/datasets/Intel/misinformation-guard-benchmark).

### Key Features:
- Fine-tuned on a synthetic dataset created for misinformation multi-label classification.
- Designed to perform well on text related to health and medicine, politics and government, climate change and environmental issues, science and technology, conspiracy theories, economics and financial markets, social and cultural issues, and technology and AI.

### Use Cases:
- Identify misinformation in text.
- Protection against adversarial attacks. 

## Description of labels
- **false**: Completely untrue or fabricated information.
- **partially true**: Contains some truth but is misleading or lacks important context.
- **mostly true**: Largely accurate but may have minor inaccuracies or omissions.
- **true**: Entirely accurate and factual information.


## Model Details
- **Training Data**: This model was trained on a synthetic dataset generated specifically for multi-label classification of text according to misinformation. 
- **Base Model**: The base model is [BERT-base](https://huggingface.co/bert-base-uncased), with 12 layers, 110M parameters.
- **Fine-tuning Process**: Fine-tuning was performed on misinformation-guard synthetic dateset for 5 epochs, using  learning rate 2e-05, with AdamW optimizer that is the default optimizer in Hugging Face `Trainer` framework with weight decay. The best hyperparameters for fine-tuning were selected by a grid search on the following grid:

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

- **Accuracy**: tbm% on the misinformation-guard synthetic dataset.
- **F1-Score**: tbm% on the misinformation-guard synthetic dataset.

- **Accuracy**: tbm% on the misinformation-guard benchmark dataset.
- **F1-Score**: tbm% on the misinformation-guard benchmark dataset.

---

## How to Use

This fine-tuned BERT model can be used for categorizing text into classes `false`, `partially true`, `mostly true`, and `true`.

### Load the model and tokenizer:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Intel/misinformation-guard")
model = AutoModelForSequenceClassification.from_pretrained("Intel/misinformation-guard")

# Example usage
inputs = tokenizer("Your input text", return_tensors="pt")
outputs = model(**inputs)
```
## Disclaimer
Misinformation Guard has been trained and validated on a limited set 
of synthetically generated data. Accuracy metrics cannot be guaranteed 
outside these narrow use cases, and therefore this tool should be 
validated within the specific context of use for which it might be deployed. 
This tool is not intended to be used to evaluate employee performance. 
This tool is not sufficient to prevent harm in many contexts, and additional
tools and techniques should be employed in any sensitive use case where 
misinformation may cause harm to individuals, communities, or society.


## Citation
If you use this model, please cite:
@misc{Intel_misinformation-guard_2024,
  author = {Murilo Gustineli https://github.com/murilogustineli},
  title = {Misinformation Guard},
  year = {2024},
  url = {https://huggingface.co/Intel/misinformation-guard},
}
