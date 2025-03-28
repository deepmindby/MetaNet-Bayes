"""
Argument list for MetaNet-Bayes

Enhanced to include all necessary parameters for training and evaluation
with adaptive gating and augmented features.
"""

import argparse
import os
import random
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="MetaNet-Bayes Arguments")

    # Base directory settings
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/zby/MetaNet-Bayes"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.expanduser("~/zby/MetaNet-Bayes/checkpoints_adaptive_gating"),
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Explicit directory for precomputed features (overrides data-location)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.expanduser("~/zby/MetaNet-Bayes/checkpoints_adaptive_gating"),
        help="Directory with trained models for evaluation",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"],
        help="Which datasets to use for training/evaluation",
    )

    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )

    # Distributed training settings
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=random.randint(10000, 20000),
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Adaptive Gating parameters
    parser.add_argument(
        "--blockwise-coef",
        action="store_true",
        default=True,
        help="Use blockwise coefficients for adaptive gating"
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.05,
        help="Base threshold for gating mechanism (αT)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for uncertainty weighting"
    )
    parser.add_argument(
        "--uncertainty-reg",
        type=float,
        default=0.01,
        help="Weight for uncertainty regularization in loss function"
    )
    parser.add_argument(
        "--num-task-vectors",
        type=int,
        default=8,
        help="Number of task vectors to simulate"
    )
    parser.add_argument(
        "--no-gating",
        action="store_true",
        default=False,
        help="Disable gating mechanism, set beta and base threshold to minimal values"
    )

    # Special evaluation options
    parser.add_argument(
        "--compare-models",
        action="store_true",
        default=False,
        help="Compare different model variations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Provide detailed output during evaluation"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with additional logging"
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    parsed_args.use_augmentation = True

    return parsed_args