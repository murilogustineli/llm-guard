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
**Domain**: Multi-label classification of text into `false`, `partially true`, `mostly true`, and `true` categories

## Dataset Description
This dataset was manually annotated to provide a comprehensive benchmark for measuring the accuracy of text classification of language models according to misinformation. 

Each example contains:
- **text**: The text input (string)
- **label**: The classification label (category: `false`, `partially true`, `mostly true`, and `true`)

## Description of labels
- **false**: Completely untrue or fabricated information.
- **partially true**: Contains some truth but is misleading or lacks important context.
- **mostly true**: Largely accurate but may have minor inaccuracies or omissions.
- **true**: Entirely accurate and factual information.

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("Intel/misinformation-guard-benchmark")
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
