"""
model.py – LightningModule for sequence-classification fine-tuning.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torchmetrics
import lightning as L
from torchmetrics.classification import F1Score
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)



class LightningModel(L.LightningModule):
    """
    PyTorch Lightning model class for fine-tuning a language model on a classification task.

    Args:
        model (AutoModelForSequenceClassification): The pre-trained base model.
        num_training_steps (int): The number of training samples times max_epochs for the learning rate scheduler.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        num_labels (int): The number of output labels for the classification task.
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        num_training_steps: int,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_labels: int = 4,
    ):
        """
        Initialize the model for sequence classification with hyperparameters and metrics.

        Args:
            model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
            num_training_steps (int): The total number of training steps.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 5e-5.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.01.
            num_labels (int, optional): The number of labels for classification. Defaults to 4.

        Attributes:
            model: The pre-trained model for sequence classification.
            num_training_steps: The total number of training steps.
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
            num_labels: The number of labels for classification.
            val_f1: The F1 score metric for validation, weighted by class.
            test_f1: The F1 score metric for test, weighted by class.
            val_acc: The accuracy metric for validation.
            test_acc: The accuracy metric for test.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_labels = num_labels

        self.val_f1 = F1Score(
            num_classes=self.num_labels, task="multiclass", average="weighted"
        )
        self.test_f1 = F1Score(
            num_classes=self.num_labels, task="multiclass", average="weighted"
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_labels
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor): Tensor indicating the attention mask for padding tokens.
            labels (Optional[torch.Tensor], optional): Labels for computing loss. Default is None.

        Returns:
            Dict: Output of the model.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def _shared_step(self, batch: Dict, stage: str) -> Optional[torch.Tensor]:
        """
        A shared step function for training, validation, and testing.

        Args:
            batch (Dict): A batch of data containing 'input_ids', 'attention_mask', and 'label'.
            stage (str): One of "train", "val", or "test" to indicate the current phase.

        Returns:
            Optional[torch.Tensor]: Returns the loss tensor for training; None for validation or testing.
        """
        outputs = self(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        logits = outputs["logits"]
        loss = outputs["loss"]
        labels = batch["label"]

        if stage == "train":
            self.log("train_loss", loss)
            return loss

        if stage == "val":
            self.val_acc(logits, labels)
            self.val_f1(logits, labels)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_f1", self.val_f1, prog_bar=True)

        if stage == "test":
            self.test_acc(logits, labels)
            self.test_f1(logits, labels)
            self.log("test_acc", self.test_acc, prog_bar=True)
            self.log("test_f1", self.test_f1, prog_bar=True)

        return None

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch (Dict): A batch of data, containing input tensors and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value for this step that is sent to the optimizer.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """
        Perform a validation step.

        Args:
            batch (Dict): A batch of validation data.
            batch_idx (int): The index of the batch.
        """
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """
        Perform a test step.

        Args:
            batch (Dict): A batch of test data.
            batch_idx (int): The index of the batch.
        """
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> Tuple[List[torch.optim.AdamW], List[Dict]]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[torch.optim.AdamW], List[Dict]]: A tuple containing a list of the optimizer(s)
            and a list of learning rate scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        num_warmup_steps = int(0.1 * self.num_training_steps)
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, self.num_training_steps
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
