"""Model evaluation utilities -- single-model eval and full suite runner."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_name: str = "Model",
) -> Tuple[float, float]:
    """Evaluate *model* on *dataloader* and return ``(avg_loss, accuracy_%)``.

    Parameters
    ----------
    model : nn.Module
        Trained model (moved to *device* automatically).
    dataloader : DataLoader
        Test / validation data.
    device : str
        ``"cuda"`` or ``"cpu"``.
    model_name : str
        Label for log output.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    print(f"  Evaluating {model_name} ...")

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = (correct / total) * 100

    print(f"    Loss: {avg_loss:.4f}  |  Acc: {avg_acc:.2f}%")
    return avg_loss, avg_acc


def run_evaluation_suite(
    models: Dict[str, nn.Module],
    test_loaders: Dict[str, DataLoader],
    device: str,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Evaluate every model on every test loader.

    Parameters
    ----------
    models : dict
        ``{model_name: trained_model}``
    test_loaders : dict
        ``{test_set_name: DataLoader}``  (e.g. ``"test_standard"``, ``"test_rotated"``)
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    Nested dict ``results[test_set_name][model_name] = (loss, accuracy)``.
    """
    results: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for test_name, loader in test_loaders.items():
        print(f"\n--- Evaluation on: {test_name} ---")
        results[test_name] = {}
        for model_name, model in models.items():
            loss, acc = evaluate_model(model, loader, device, model_name)
            results[test_name][model_name] = (loss, acc)

    return results
