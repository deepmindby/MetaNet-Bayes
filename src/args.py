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
        help="The type of model (e.g. RN50, RN101, ViT-B-32). Supports ResNet50 (RN50), ResNet101 (RN101) and ViT models.",
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
        default=0.0005,
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
        default=10,
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
    parser.add_argument(
        "--no-metanet",
        action="store_true",
        default=False,
        help="Remove MetaNet and use original Atlas implementation with precomputed features"
    )
    parser.add_argument(
        "--finetuning-mode",
        type=str,
        default="standard",
        choices=["standard", "linear"],
        help="Model finetuning mode: 'standard' or 'linear'",
    )
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        default=50.0,
        help="Learning rate multiplier for gating parameters"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay for gating parameters"
    )
    parser.add_argument(
        "--reg-coefficient",
        type=float,
        default=0.001,
        help="Regularization coefficient for beta and threshold"
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=0.0001,
        help="Weight for margin loss"
    )
    parser.add_argument(
        "--gating-no-metanet",
        action="store_true",
        default=False,
        help="Apply gating mechanism directly to Atlas (no-metanet) model"
    )
    # Variational inference parameters
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.01,
        help="Weight for KL divergence in ELBO"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of MC samples during training"
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=20,
        help="Number of MC samples during evaluation"
    )
    parser.add_argument(
        "--save-uncertainty",
        action="store_true",
        default=False,
        help="Save uncertainty analysis plots and data"
    )
    parser.add_argument(
        "--detailed-analysis",
        action="store_true",
        default=False,
        help="Perform detailed feature importance analysis"
    )
    parser.add_argument(
        "--variational-gating",
        action="store_true",
        default=True,
        help="Enable variational gating mechanism"
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    parsed_args.use_augmentation = True

    return parsed_args