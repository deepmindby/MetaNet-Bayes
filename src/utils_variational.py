"""
Utility functions for variational inference training and evaluation.

This module provides helper functions for visualization, uncertainty
metrics, and other utilities specific to variational inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import os
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_uncertainty_histogram(uncertainties, values=None, title="Uncertainty Distribution"):
    """Plot histogram of uncertainty values.

    Parameters:
    ----------
    uncertainties: torch.Tensor or np.ndarray
        Uncertainty values
    values: torch.Tensor or np.ndarray, optional
        Values corresponding to uncertainties (e.g., coefficients)
    title: str
        Plot title

    Returns:
    ----------
    fig: matplotlib.figure.Figure
        Figure object
    """
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.detach().cpu().numpy().flatten()
    else:
        uncertainties = uncertainties.flatten()

    fig, ax = plt.subplots(figsize=(10, 6))

    if values is not None:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().flatten()
        else:
            values = values.flatten()

        # Create a custom colormap
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_colormap",
            [(0, "blue"), (0.5, "white"), (1, "red")],
            N=256
        )

        # Use a 2D histogram to show relationship
        h = ax.hist2d(
            uncertainties,
            values,
            bins=50,
            cmap=custom_cmap,
            norm=None
        )
        fig.colorbar(h[3], ax=ax, label="Count")
        ax.set_ylabel("Coefficient Value")
    else:
        # Simple 1D histogram
        ax.hist(uncertainties, bins=50, alpha=0.7, color="blue")
        ax.set_ylabel("Count")

    ax.set_xlabel("Uncertainty")
    ax.set_title(title)

    return fig


def plot_reliability_diagram(y_true, y_prob, n_bins=10, title="Reliability Diagram"):
    """Plot reliability diagram for probabilistic predictions.

    Parameters:
    ----------
    y_true: np.ndarray
        Ground truth labels (binary or one-hot encoded)
    y_prob: np.ndarray
        Predicted probabilities
    n_bins: int
        Number of bins for reliability curve
    title: str
        Plot title

    Returns:
    ----------
    fig: matplotlib.figure.Figure
        Figure object
    """
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Calculate Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Create reliability diagram
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"Model (Brier: {brier:.3f})")

    # Plot histogram of predicted probabilities
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=n_bins, alpha=0.3, color="blue", range=(0, 1))
    ax2.set_ylabel("Count")
    ax2.grid(False)

    # Set labels and title
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="best")

    return fig


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE).

    Parameters:
    ----------
    y_true: np.ndarray
        Ground truth labels (one-hot encoded)
    y_prob: np.ndarray
        Predicted probabilities
    n_bins: int
        Number of bins for ECE calculation

    Returns:
    ----------
    ece: float
        Expected Calibration Error
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    # Get predicted class and confidence
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        # Multi-class case
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)

        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # Convert one-hot to class indices
            y_true = np.argmax(y_true, axis=1)
    else:
        # Binary case
        confidences = y_prob
        predictions = (y_prob >= 0.5).astype(np.int32)

    # Create bins
    bin_indices = np.linspace(0, 1, n_bins + 1)
    bin_indices[-1] = 1.0001  # Ensure all samples fall into bins

    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences >= bin_indices[i]) & (confidences < bin_indices[i + 1])
        if np.sum(bin_mask) > 0:
            bin_confidence = np.mean(confidences[bin_mask])
            bin_accuracy = np.mean(predictions[bin_mask] == y_true[bin_mask])
            bin_size = np.sum(bin_mask) / len(confidences)

            ece += bin_size * np.abs(bin_accuracy - bin_confidence)

    return ece


def get_uncertainty_metrics(predictions_dict, labels):
    """Calculate uncertainty-related metrics.

    Parameters:
    ----------
    predictions_dict: dict
        Dictionary with prediction statistics from monte_carlo_predictions
    labels: torch.Tensor
        Ground truth labels

    Returns:
    ----------
    metrics: dict
        Dictionary with uncertainty metrics
    """
    # Convert to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = predictions_dict["predictions"].numpy()
    mean_probs = predictions_dict["mean_probs"].numpy()
    predictive_entropy = predictions_dict["predictive_entropy"].numpy()
    aleatoric_uncertainty = predictions_dict["aleatoric_uncertainty"].numpy()
    epistemic_uncertainty = predictions_dict["epistemic_uncertainty"].numpy()

    # Calculate accuracy
    accuracy = np.mean(predictions == labels)

    # Calculate ECE
    ece = expected_calibration_error(labels, mean_probs)

    # Separate correct and incorrect predictions
    correct_mask = (predictions == labels)
    incorrect_mask = ~correct_mask

    # Calculate uncertainty metrics for correct and incorrect predictions
    metrics = {
        "accuracy": accuracy,
        "ece": ece,
        "avg_predictive_entropy": np.mean(predictive_entropy),
        "avg_aleatoric_uncertainty": np.mean(aleatoric_uncertainty),
        "avg_epistemic_uncertainty": np.mean(epistemic_uncertainty),
    }

    # Only calculate if there are both correct and incorrect predictions
    if np.any(correct_mask) and np.any(incorrect_mask):
        metrics.update({
            "correct_predictive_entropy": np.mean(predictive_entropy[correct_mask]),
            "incorrect_predictive_entropy": np.mean(predictive_entropy[incorrect_mask]),
            "correct_epistemic_uncertainty": np.mean(epistemic_uncertainty[correct_mask]),
            "incorrect_epistemic_uncertainty": np.mean(epistemic_uncertainty[incorrect_mask]),
            "uncertainty_auroc": calculate_uncertainty_auroc(epistemic_uncertainty, correct_mask),
        })

    return metrics


def calculate_uncertainty_auroc(uncertainties, correct_mask):
    """Calculate AUROC for using uncertainty to detect errors.

    Parameters:
    ----------
    uncertainties: np.ndarray
        Uncertainty values
    correct_mask: np.ndarray
        Boolean mask of correct predictions

    Returns:
    ----------
    auroc: float
        AUROC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        # Higher uncertainty should predict incorrect classifications
        return roc_auc_score(~correct_mask, uncertainties)
    except:
        return 0.0


def visualize_posterior_distribution(posterior_stats, num_display=5, num_task_vectors=8, blockwise=False):
    """Visualize posterior distribution statistics.

    Parameters:
    ----------
    posterior_stats: dict
        Dictionary with posterior statistics from get_posterior_stats
    num_display: int
        Number of samples to display
    num_task_vectors: int
        Number of task vectors
    blockwise: bool
        Whether using blockwise coefficients

    Returns:
    ----------
    figs: list
        List of matplotlib figures
    """
    means = posterior_stats["means"]
    stdevs = posterior_stats["stdevs"]
    coeff_var = posterior_stats["coeff_var"]
    inclusion_probs = posterior_stats["inclusion_probs"]
    binary_indicators = posterior_stats.get("binary_indicators", None)

    batch_size = means.shape[0]
    num_display = min(num_display, batch_size)

    figs = []

    # For each sample
    for i in range(num_display):
        if blockwise:
            # Reshape for blockwise visualization
            sample_means = means[i].reshape(num_task_vectors, -1)
            sample_stdevs = stdevs[i].reshape(num_task_vectors, -1)
            sample_inclusion_probs = inclusion_probs[i].reshape(num_task_vectors, -1)
            if binary_indicators is not None:
                sample_binary_indicators = binary_indicators[i].reshape(num_task_vectors, -1)
            else:
                sample_binary_indicators = None

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Sample {i + 1} Posterior Distribution")

            # Plot means
            im0 = axes[0, 0].imshow(sample_means, aspect='auto', cmap='coolwarm')
            axes[0, 0].set_title("Coefficient Means")
            axes[0, 0].set_ylabel("Task Vector")
            axes[0, 0].set_xlabel("Block")
            fig.colorbar(im0, ax=axes[0, 0])

            # Plot standard deviations
            im1 = axes[0, 1].imshow(sample_stdevs, aspect='auto', cmap='viridis')
            axes[0, 1].set_title("Coefficient Std Deviations")
            axes[0, 1].set_ylabel("Task Vector")
            axes[0, 1].set_xlabel("Block")
            fig.colorbar(im1, ax=axes[0, 1])

            # Plot inclusion probabilities
            im2 = axes[1, 0].imshow(sample_inclusion_probs, aspect='auto', cmap='plasma')
            axes[1, 0].set_title("Inclusion Probabilities")
            axes[1, 0].set_ylabel("Task Vector")
            axes[1, 0].set_xlabel("Block")
            fig.colorbar(im2, ax=axes[1, 0])

            # Plot binary indicators or coefficient of variation
            if sample_binary_indicators is not None:
                im3 = axes[1, 1].imshow(sample_binary_indicators, aspect='auto', cmap='magma')
                axes[1, 1].set_title("Binary Indicators")
            else:
                sample_coeff_var = coeff_var[i].reshape(num_task_vectors, -1)
                im3 = axes[1, 1].imshow(sample_coeff_var, aspect='auto', cmap='magma')
                axes[1, 1].set_title("Coefficient of Variation")
            axes[1, 1].set_ylabel("Task Vector")
            axes[1, 1].set_xlabel("Block")
            fig.colorbar(im3, ax=axes[1, 1])
        else:
            # For non-blockwise, just show bar plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Sample {i + 1} Posterior Distribution")

            x = np.arange(num_task_vectors)
            width = 0.35

            # Plot means with standard deviation as error bars
            axes[0, 0].bar(x, means[i], yerr=stdevs[i], alpha=0.7)
            axes[0, 0].set_title("Coefficient Posterior Distribution")
            axes[0, 0].set_ylabel("Value")
            axes[0, 0].set_xlabel("Task Vector")

            # Plot coefficient of variation
            axes[0, 1].bar(x, coeff_var[i], alpha=0.7, color='orange')
            axes[0, 1].set_title("Coefficient of Variation (Relative Uncertainty)")
            axes[0, 1].set_ylabel("Value")
            axes[0, 1].set_xlabel("Task Vector")

            # Plot inclusion probabilities
            axes[1, 0].bar(x, inclusion_probs[i], alpha=0.7, color='green')
            axes[1, 0].set_title("Inclusion Probabilities")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].set_xlabel("Task Vector")

            # Plot binary indicators if available, otherwise something else
            if binary_indicators is not None:
                axes[1, 1].bar(x, binary_indicators[i], alpha=0.7, color='purple')
                axes[1, 1].set_title("Binary Indicators")
            else:
                # Just duplicate inclusion probabilities as a fallback
                axes[1, 1].bar(x, inclusion_probs[i], alpha=0.7, color='purple')
                axes[1, 1].set_title("Inclusion Probabilities (duplicate)")
            axes[1, 1].set_ylabel("Value")
            axes[1, 1].set_xlabel("Task Vector")

        plt.tight_layout()
        figs.append(fig)

    return figs


def save_uncertainty_analysis(predictions_dict, labels, save_dir, prefix="uncertainty"):
    """Save uncertainty analysis plots and metrics.

    Parameters:
    ----------
    predictions_dict: dict
        Dictionary with prediction statistics from monte_carlo_predictions
    labels: torch.Tensor
        Ground truth labels
    save_dir: str
        Directory to save outputs
    prefix: str
        Prefix for saved files

    Returns:
    ----------
    metrics: dict
        Dictionary with uncertainty metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    # Calculate metrics
    metrics = get_uncertainty_metrics(predictions_dict, labels)

    # Save metrics to file
    metrics_path = os.path.join(save_dir, f"{prefix}_metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # Convert tensors to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = predictions_dict["predictions"].numpy()
    mean_probs = predictions_dict["mean_probs"].numpy()
    predictive_entropy = predictions_dict["predictive_entropy"].numpy()
    epistemic_uncertainty = predictions_dict["epistemic_uncertainty"].numpy()

    # Separate correct and incorrect predictions
    correct_mask = (predictions == labels)

    # Plot reliability diagram
    if mean_probs.shape[1] > 2:  # Multi-class
        # One-vs-rest reliability diagrams for top 3 classes
        for i in range(min(3, mean_probs.shape[1])):
            fig = plot_reliability_diagram(
                (labels == i).astype(np.int32),
                mean_probs[:, i],
                title=f"Reliability Diagram (Class {i})"
            )
            fig.savefig(os.path.join(save_dir, f"{prefix}_reliability_class{i}.png"), dpi=300)
            plt.close(fig)
    else:  # Binary
        fig = plot_reliability_diagram(
            labels,
            mean_probs[:, 1] if mean_probs.shape[1] > 1 else mean_probs,
            title="Reliability Diagram"
        )
        fig.savefig(os.path.join(save_dir, f"{prefix}_reliability.png"), dpi=300)
        plt.close(fig)

    # Plot uncertainty distributions
    if np.any(correct_mask) and np.any(~correct_mask):
        # Plot epistemic uncertainty distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(epistemic_uncertainty[correct_mask], bins=20, alpha=0.5, label="Correct")
        ax.hist(epistemic_uncertainty[~correct_mask], bins=20, alpha=0.5, label="Incorrect")
        ax.set_xlabel("Epistemic Uncertainty")
        ax.set_ylabel("Count")
        ax.set_title("Epistemic Uncertainty Distribution by Correctness")
        ax.legend()
        fig.savefig(os.path.join(save_dir, f"{prefix}_epistemic_uncertainty.png"), dpi=300)
        plt.close(fig)

        # Plot predictive entropy distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(predictive_entropy[correct_mask], bins=20, alpha=0.5, label="Correct")
        ax.hist(predictive_entropy[~correct_mask], bins=20, alpha=0.5, label="Incorrect")
        ax.set_xlabel("Predictive Entropy")
        ax.set_ylabel("Count")
        ax.set_title("Predictive Entropy Distribution by Correctness")
        ax.legend()
        fig.savefig(os.path.join(save_dir, f"{prefix}_predictive_entropy.png"), dpi=300)
        plt.close(fig)

    return metrics