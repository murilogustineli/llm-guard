"""
data.py – data loading utilities and the custom Dataset for misinformation-guard fine-tuning.
"""

import torch
import pandas as pd
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame and performs checks to ensure
    it has the necessary columns and that the label column is numeric.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If the required columns ('text' and 'label') are missing.
        ValueError: If the 'label' column is not numeric.
        Exception: If any other error occurs when parsing the file.
    """
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(file_path)

        # Check if 'text' and 'label' columns exist
        if "text" not in df.columns or "label" not in df.columns:
            raise KeyError(
                f"The CSV file '{file_path}' must contain 'text' and 'label' columns."
            )

        # Check if the 'label' column is numeric
        if not pd.api.types.is_numeric_dtype(df["label"]):
            raise ValueError(
                f"The 'label' column in '{file_path}' must be numeric for fine-tuning."
            )

        return df

    except FileNotFoundError as e:
        print(f"Error: The file '{file_path}' was not found - {e}")
        raise
    except KeyError as e:
        print(f"Error: {e}")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error: There was an issue parsing the file '{file_path}' - {e}")
        raise


class TextDataset(Dataset):
    """
    Custom dataset for text classification. The dataset can accept either a pandas DataFrame or a list of strings.
    - If a DataFrame is provided, it should have columns "text" (input text) and "label" (numeric labels).
    - If a list of strings is provided, it will be used as the text data, and labels will be None.

    Args:
        data (Union[pd.DataFrame, List[str]]): The input data. If a DataFrame, it should have 'text' and 'label' columns.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the text.
        max_length (int, optional): The maximum length for tokenized sequences. Default is 512.

    Attributes:
        data (Union[pd.DataFrame, List[str]]): The input data.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        max_length (int): The maximum length for tokenization.
        is_dataframe (bool): A flag indicating whether the input data is a DataFrame.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize the class with data, tokenizer, and optional max_length.

        Args:
            data (Union[pd.DataFrame, List[str]]): The input data, either as a pandas DataFrame or a list of strings.
            tokenizer (PreTrainedTokenizer): The tokenizer used for processing text data.
            max_length (int, optional): The maximum length for tokenized sequences. Defaults to 512.

        Attributes:
            data: The input data, either a pandas DataFrame or a list of strings.
            tokenizer: The tokenizer for processing text.
            max_length: The maximum length for tokenized sequences.
            is_dataframe (bool): A flag indicating whether the data is a pandas DataFrame (True) or a list of strings (False).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_dataframe = hasattr(data, "iloc")  # Check if the data is a dataframe

    def __len__(self) -> int:
        """
        Return the number of elements in the data.

        Returns:
            int: The length of the dataframe or list.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tokenize and return the input text and corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'attention_mask', and 'label' (if applicable).
        """
        if self.is_dataframe:
            row = self.data.iloc[idx]
            text = row["text"]
            label = row["label"]
        else:
            text = self.data[idx]
            label = None

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)

        return item


def prepare_data(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares data for training, validation, and testing by loading CSV files and creating corresponding datasets
    and dataloaders.

    Args:
        train_path (str): Path to the training CSV file.
        val_path (str): Path to the validation CSV file.
        test_path (str): Path to the testing CSV file.
        tokenizer (PreTrainedTokenizer): A tokenizer instance used for processing text data.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the training, validation, and testing dataloaders.

    Raises:
        FileNotFoundError: If any of the CSV files do not exist.
        KeyError: If the CSV files are missing required columns ('text' and 'label').
        ValueError: If the 'label' column is not numeric.
        Exception: If any other error occurs when parsing the files or creating the datasets/dataloaders.
    """
    # Load CSV files
    train_df = load_csv(train_path)
    val_df = load_csv(val_path)
    test_df = load_csv(test_path)

    # Create datasets and dataloaders
    try:
        train_dataset = TextDataset(train_df, tokenizer)
        val_dataset = TextDataset(val_df, tokenizer)
        test_dataset = TextDataset(test_df, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
    except Exception as e:
        print(f"An error occurred in preparing dataloaders: {e}")
        raise
    return train_loader, val_loader, test_loader
