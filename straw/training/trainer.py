"""Unified training loop for any model (standalone, ResNet, modulator, ...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    """Container for training history returned by :meth:`Trainer.train`."""

    model: nn.Module
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)


class Trainer:
    """Single training loop that works for every model type.

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.
    lr : float
        Learning rate for Adam.
    batch_size : int
        Mini-batch size.
    """

    def __init__(self, device: str, lr: float = 1e-4, batch_size: int = 128):
        self.device = device
        self.lr = lr
        self.batch_size = batch_size

    def train(
        self,
        model: nn.Module,
        train_dataset,
        num_epochs: int,
        model_name: str = "Model",
    ) -> TrainResult:
        """Run the training loop and return a :class:`TrainResult`.

        Parameters
        ----------
        model : nn.Module
            Already on the correct device.
        train_dataset
            A PyTorch ``Dataset`` (or ``Subset``).
        num_epochs : int
            Number of full passes over the dataset.
        model_name : str
            Label used in log output.
        """
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
        )

        print(f"\n[{model_name}] Starting training ({num_epochs} epochs) ...")
        loss_history: List[float] = []
        accuracy_history: List[float] = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = (correct / total) * 100

            loss_history.append(epoch_loss)
            accuracy_history.append(epoch_acc)

            print(
                f"  Epoch {epoch + 1}/{num_epochs}  "
                f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%"
            )

        return TrainResult(
            model=model,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
        )
