#!/usr/bin/env python3
"""STRAW experiment runner -- trains & evaluates all models from a YAML config.

Usage
-----
Run all experiments::

    python run_experiment.py

Run a single experiment by name::

    python run_experiment.py --experiment full_rank16

Use a custom config file::

    python run_experiment.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from straw.data import get_datasets, get_dataloaders
from straw.evaluation import evaluate_model, run_evaluation_suite
from straw.models import MODEL_REGISTRY
from straw.training import Trainer
from straw.visualization.plots import (
    plot_comparison,
    plot_evaluation_suite,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_with_defaults(experiment: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an experiment block with the defaults (experiment wins)."""
    merged = {**defaults, **experiment}
    merged["device"] = resolve_device(merged.get("device", "auto"))
    return merged


# ──────────────────────────────────────────────
# Single experiment
# ──────────────────────────────────────────────

def run_single_experiment(cfg: Dict[str, Any], output_root: str = "outputs") -> None:
    """Execute one complete experiment: train all models, evaluate, plot."""

    name = cfg["name"]
    device = cfg["device"]
    seed = cfg.get("seed", 42)
    set_seed(seed)

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT: {name}")
    print("=" * 60)
    print(f"  device={device}  lr={cfg['lr']}  batch_size={cfg['batch_size']}")
    print(f"  rank={cfg['rank']}  train_subset={cfg['train_subset']}  epochs={cfg['num_epochs']}")
    print("=" * 60)

    # Output directory
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Data ──────────────────────────────────
    datasets = get_datasets(
        data_root=cfg.get("data_root", "./data"),
        train_subset=cfg["train_subset"],
    )
    loaders = get_dataloaders(datasets, batch_size=cfg["batch_size"])

    # Separate train and test loaders
    train_ds = datasets["train"]
    test_loaders = {k: v for k, v in loaders.items() if k != "train"}

    # ── Train ─────────────────────────────────
    trainer = Trainer(device=device, lr=cfg["lr"], batch_size=cfg["batch_size"])
    trained_models: Dict[str, torch.nn.Module] = {}
    all_losses = []
    all_accuracies = []
    model_names = []

    for model_key in cfg.get("models", ["standalone", "resnet34", "modulator"]):
        if model_key not in MODEL_REGISTRY:
            print(f"  [WARN] Unknown model '{model_key}', skipping.")
            continue

        set_seed(seed)  # reset seed so each model gets the same data order
        model = MODEL_REGISTRY[model_key](cfg)
        result = trainer.train(
            model=model,
            train_dataset=train_ds,
            num_epochs=cfg["num_epochs"],
            model_name=model_key,
        )

        trained_models[model_key] = result.model
        all_losses.append(result.loss_history)
        all_accuracies.append(result.accuracy_history)
        model_names.append(model_key)

        # Save checkpoint
        ckpt_path = os.path.join(out_dir, f"{model_key}.pt")
        torch.save(result.model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # ── Training curves ───────────────────────
    if all_losses:
        plot_comparison(
            all_losses, all_accuracies, model_names,
            save_path=os.path.join(out_dir, "training_curves.png"),
        )

    # ── Evaluate ──────────────────────────────
    if trained_models and test_loaders:
        results = run_evaluation_suite(trained_models, test_loaders, device)
        plot_evaluation_suite(results, save_dir=out_dir)

    print(f"\n  Experiment '{name}' complete. Outputs in {out_dir}/\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="STRAW experiment runner")
    parser.add_argument(
        "--config", type=str, default="configs/experiment.yaml",
        help="Path to YAML config file (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Run only the experiment with this name (default: run all)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Root directory for outputs (default: outputs/)",
    )
    args = parser.parse_args()

    raw_config = load_config(args.config)
    defaults = raw_config.get("defaults", {})
    experiments = raw_config.get("experiments", [])

    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            print(f"Error: no experiment named '{args.experiment}' in config.")
            return

    print(f"Running {len(experiments)} experiment(s) from {args.config}")

    for exp in experiments:
        cfg = merge_with_defaults(exp, defaults)
        run_single_experiment(cfg, output_root=args.output)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
