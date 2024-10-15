---
task_categories:
- text-classification
languages:
- en
license: TBD (apache-2.0 ? GPL)
---

# Benchmark Text Classification Dataset

**Dataset type**: Manually Annotated  
**Number of samples**: 500  
**Task**: Text Classification  
**Domain**: Multi-label classification of text into polite, somewhat polite, neutral, and impolite categories

## Dataset Description
This dataset was manually annotated to provide a comprehensive benchmark for measuring the accuracy of text classification of language models according to politeness. Data has been collected from the following resources with all personal identifiers removed.

- [Yelp customer reviews](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset)
- [Airline tweets](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data)
- [Customer support on twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- [Comments from “Ratings and Comments” section of Intel Saba trainings](https://intel.sabacloud.com)



Each example contains:

- **text**: The text input (string)
- **label**: The classification label (category: polite, somewhat polite, neutral, and impolite)

## Description of labels
- **polite**: Text is considerate and shows respect and good manners, often including courteous phrases and a friendly tone.
- **somewhat polite**: Text is generally respectful but lacks warmth or formality, communicating with a decent level of courtesy.
- **neutral**: Text is straightforward and factual, without emotional undertones or specific attempts at politeness.
- **impolite**: Text is disrespectful or rude, often blunt or dismissive, showing a lack of consideration for the recipient's feelings.

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("Intel/polite-guard-benchmark")
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