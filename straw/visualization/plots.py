"""Plotting utilities for training curves and evaluation bar charts."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
    losses: List[List[float]],
    accuracies: List[List[float]],
    model_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side training loss and accuracy curves.

    Parameters
    ----------
    losses : list of lists
        Per-epoch loss history for each model.
    accuracies : list of lists
        Per-epoch accuracy history for each model.
    model_names : list of str
        Legend labels.
    save_path : str, optional
        If given, save the figure to this path instead of showing it.
    """
    epochs = range(1, len(losses[0]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for loss, name in zip(losses, model_names):
        ax1.plot(epochs, loss, label=name, marker="o", linestyle="-")
    ax1.set_title("Training Loss Comparison", fontsize=14)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    for acc, name in zip(accuracies, model_names):
        ax2.plot(epochs, acc, label=name, marker="s", linestyle="-")
    ax2.set_title("Training Accuracy Comparison", fontsize=14)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training curves to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_evaluation_bars(
    losses: List[float],
    accuracies: List[float],
    model_names: List[str],
    title: str = "Model Evaluation on Unseen Test Data",
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side bar charts for test loss and accuracy.

    Parameters
    ----------
    losses : list of float
        One loss value per model.
    accuracies : list of float
        One accuracy (%) value per model.
    model_names : list of str
        Bar labels.
    title : str
        Super-title for the figure.
    save_path : str, optional
        If given, save the figure to this path instead of showing it.
    """
    x_pos = np.arange(len(model_names))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    # Extend palette if more than 3 models
    while len(colors) < len(model_names):
        colors.append(f"#{np.random.randint(0, 0xFFFFFF):06x}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy bars
    bars1 = ax1.bar(x_pos, accuracies, color=colors[: len(model_names)], alpha=0.8)
    ax1.set_title("Final Test Accuracy (Higher is Better)", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, h,
            f"{h:.2f}%", ha="center", va="bottom", fontweight="bold",
        )

    # Loss bars
    bars2 = ax2.bar(x_pos, losses, color=colors[: len(model_names)], alpha=0.8)
    ax2.set_title("Final Test Loss (Lower is Better)", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names)
    ax2.set_ylabel("Loss")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, h,
            f"{h:.4f}", ha="center", va="bottom", fontweight="bold",
        )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved evaluation bars to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_evaluation_suite(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    save_dir: Optional[str] = None,
) -> None:
    """Plot bar charts for every test set in an evaluation suite result.

    Parameters
    ----------
    results : dict
        Output of :func:`straw.evaluation.run_evaluation_suite`.
        ``results[test_set_name][model_name] = (loss, acc)``
    save_dir : str, optional
        Directory to save figures. If ``None``, figures are shown interactively.
    """
    import os

    for test_name, model_results in results.items():
        names = list(model_results.keys())
        losses = [model_results[n][0] for n in names]
        accs = [model_results[n][1] for n in names]

        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"eval_{test_name}.png")

        plot_evaluation_bars(
            losses, accs, names,
            title=f"Evaluation: {test_name}",
            save_path=save_path,
        )
