"""
This script fine-tunes a pre-trained language model for text classification using PyTorch Lightning.
It was inspired by the following URL: https://github.com/intel/polite-guard/blob/main/fine-tuner/fine-tune.py
"""

import os
import argparse
import lightning as L

from datetime import datetime
from .data import prepare_data
from .model import LightningModel
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def parse_devices(value: str) -> str | int:
    """
    Parses the devices argument for the number of devices to use for training.

    The argument can either be an integer, representing the number of devices,
    or the string 'auto', which automatically selects the available devices.

    Args:
        value (str): The string representation of the argument value.

    Returns:
        str | int: Returns 'auto' if the value is 'auto', or an integer if
        the value is a valid integer.

    Raises:
        argparse.ArgumentTypeError: If the value is neither 'auto' nor a valid integer.
    """
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device value: {value}. It must be an integer or 'auto'."
        )


def validate_positive_integer(value: str) -> int:
    """
    Validate that the input is a positive integer.

    Args:
        value: The input string from argparse

    Returns:
        int: The validated integer value

    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError(
                f"The input value must be positive, got {int_value}"
            )
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


def validate_positive_float(value: str) -> float:
    """
    Validate that the input is a positive float.

    Args:
        value: The input string from argparse

    Returns:
        float: The validated float value

    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    try:
        float_value = float(value)
        if float_value <= 0:
            raise argparse.ArgumentTypeError(
                f"The input value must be positive, got {float_value}"
            )
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model for text classification."
    )

    # Model parameters
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained base model checkpoint",
    )
    parser.add_argument(
        "--num_labels",
        type=validate_positive_integer,
        default=4,
        help="Number of labels in classification task",
    )

    # Data parameters
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to the training CSV file"
    )
    parser.add_argument(
        "--val-data", type=str, required=True, help="Path to the validation CSV file"
    )
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to the test CSV file"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=validate_positive_integer,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=validate_positive_float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=validate_positive_float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--max_epochs",
        type=validate_positive_integer,
        default=2,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--patience",
        type=validate_positive_integer,
        default=3,
        help="The number of epochs with no improvement in the monitored metric after which training will be stopped.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="The minimum change in the monitored metric to qualify as an improvement.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Precision for training (e.g., '16-mixed')",
    )
    parser.add_argument(
        "--num_workers",
        type=validate_positive_integer,
        default=6,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "hpu", "tpu", "mps", "auto"],
        default="auto",
        help="Type of accelerator to use for training. Options include 'cpu', 'gpu', 'hpu', 'tpu', 'mps', or 'auto' to automatically select the available hardware.",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default="auto",
        help="Number of devices to use for training. Can be an integer or 'auto' to automatically select the available devices.",
    )

    # Logging and checkpointing parameters
    parser.add_argument(
        "--logger",
        type=str,
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="Logging framework to use. Options are 'tensorboard' or 'wandb'. Default is 'tensorboard'.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for saving logs"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment"
    )

    args = parser.parse_args()

    # Default experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-lr-{args.learning_rate}-bs-{args.batch_size}"

    return args


def main() -> None:
    """
    Train and test the model with user-specified arguments.
    """

    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

    train_loader, val_loader, test_loader = prepare_data(
        args.train_data,
        args.val_data,
        args.test_data,
        tokenizer,
        args.batch_size,
        args.num_workers,
    )

    num_training_steps = len(train_loader) * args.max_epochs

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt, num_labels=args.num_labels
    )
    lightning_model = LightningModel(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_labels=args.num_labels,
        num_training_steps=num_training_steps,
    )

    # Setup callbacks and logger
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_f1"),
        EarlyStopping(
            monitor="val_f1",
            patience=args.patience,
            min_delta=args.min_delta,
            mode="max",
            verbose=True,
        ),
    ]
    if args.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)
    elif args.logger == "wandb":
        logger = WandbLogger(save_dir=args.log_dir, project=args.experiment_name)

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        precision=args.precision,
        devices=args.devices,
        logger=logger,
        log_every_n_steps=10,
    )

    # Train the model
    try:
        trainer.fit(
            lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    except Exception as e:
        print(f"An error occurred in fine-tuning the model: {e}")
        raise

    # Test the best model
    try:
        trainer.test(lightning_model, test_loader, ckpt_path="best")
    except Exception as e:
        print(f"An error occurred in testing the fine-tuned model: {e}")
        raise


if __name__ == "__main__":
    main()
