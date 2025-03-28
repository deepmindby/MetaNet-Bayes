"""
Argument list

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import argparse
import os
import random
import torch

def int_or_float(value):
    if '.' in value:
        return float(value)
    return int(value)

def parse_arguments():
    parser = argparse.ArgumentParser()

    def generate_random_port():
        return random.randint(10000, 20000)

    # Base parameters
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("/home/haichao/zby/MetaNet-Bayes"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"],
        help="Which datasets to use for training. If not specified, uses all datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/haichao/zby/MetaNet-Bayes/checkpoints_adaptive_gating",
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=generate_random_port(),
        help="Port for distributed training. If not specified, a random port will be assigned."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Adaptive Gating specific parameters
    parser.add_argument(
        "--blockwise-coef",
        action="store_true",
        default=False,
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
        "--use-augmentation",
        action="store_true",
        default=True,
        help="Whether to use augmented features during training"
    )
    parser.add_argument(
        "--max-augmentations",
        type=int,
        default=None,
        help="Maximum number of augmentation versions to use (None for all)"
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args