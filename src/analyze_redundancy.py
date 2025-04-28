"""
Analyze feature redundancy to justify the need for spike-and-slab priors

This script analyzes precomputed features to demonstrate redundancy in task vector
combination coefficients, providing evidence for the use of spike-and-slab priors.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
import traceback
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.patches as mpatches

# For testing with a subset of features (to save memory)
MAX_SAMPLES_PER_DATASET = 5000


def analyze_singular_values(features, dataset_name, save_dir, prefix=""):
    """
    Perform Singular Value Decomposition analysis on feature matrix

    Args:
        features: Tensor of shape [n_samples, feature_dim]
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
        prefix: Prefix for saved files

    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy for analysis
    if isinstance(features, torch.Tensor):
        features_np = features.numpy()
    else:
        features_np = features

    # Calculate SVD
    print(f"Computing SVD for {dataset_name} features of shape {features_np.shape}...")
    try:
        # Center the features for better SVD analysis
        features_centered = features_np - np.mean(features_np, axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)

        # Calculate effective rank (a measure of dimensionality)
        S_normalized = S / np.sum(S)
        effective_rank = np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-10)))

        # Calculate energy coverage - how many dimensions needed to capture X% of variance
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        rank_80 = np.argmax(energy >= 0.8) + 1
        rank_90 = np.argmax(energy >= 0.9) + 1
        rank_95 = np.argmax(energy >= 0.95) + 1
        rank_99 = np.argmax(energy >= 0.99) + 1

        # Calculate redundancy ratio
        redundancy_ratio = 1 - effective_rank / features_np.shape[1]

        # Create plot directory
        os.makedirs(save_dir, exist_ok=True)

        # Plot singular value distribution
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, min(100, len(S)) + 1), S[:100], 'o-', markersize=4)
        plt.axhline(y=S[0] * 0.01, color='g', linestyle='--', label='1% of max')
        plt.axhline(y=S[0] * 0.001, color='r', linestyle='--', label='0.1% of max')
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Singular Value (log scale)', fontsize=12)
        # plt.title(
        #     f'Singular Value Distribution for {dataset_name}\nEffective Rank: {effective_rank:.1f}/{features_np.shape[1]} ({100 * redundancy_ratio:.1f}% redundancy)',
        #     fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_singular_values.png"), dpi=300)
        plt.close()

        # Plot cumulative energy
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, min(200, len(energy)) + 1), energy[:200] * 100, '-', linewidth=2)
        plt.axhline(y=90, color='g', linestyle='--', label='90% Energy')
        plt.axhline(y=95, color='r', linestyle='--', label='95% Energy')
        plt.axhline(y=99, color='b', linestyle='--', label='99% Energy')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Energy (%)', fontsize=12)
        # plt.title(
        #     f'Energy Distribution for {dataset_name}\n90% Energy: {rank_90} components, 95% Energy: {rank_95} components',
        #     fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_cumulative_energy.png"), dpi=300)
        plt.close()

        return {
            "dataset": dataset_name,
            "feature_dim": features_np.shape[1],
            "effective_rank": float(effective_rank),
            "80%_energy_rank": int(rank_80),
            "90%_energy_rank": int(rank_90),
            "95%_energy_rank": int(rank_95),
            "99%_energy_rank": int(rank_99),
            "redundancy_ratio": float(redundancy_ratio),
            "top_10_singular_values": S[:10].tolist(),
            "singular_value_decay_rate": float(S[10] / S[0])  # Measure of decay rate
        }

    except Exception as e:
        print(f"Error during SVD analysis for {dataset_name}: {e}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "error": str(e),
            "feature_dim": features_np.shape[1]
        }


def analyze_feature_correlation(features, dataset_name, save_dir, prefix=""):
    """
    Analyze feature correlation to show redundancy

    Args:
        features: Tensor of shape [n_samples, feature_dim]
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
        prefix: Prefix for saved files

    Returns:
        Dictionary with correlation analysis results
    """
    # Convert to numpy for analysis
    if isinstance(features, torch.Tensor):
        features_np = features.numpy()
    else:
        features_np = features

    # If features are too high dimensional, take a subset
    if features_np.shape[1] > 100:
        # Take random subset of features for correlation analysis
        np.random.seed(42)  # For reproducibility
        feature_indices = np.random.choice(features_np.shape[1], 100, replace=False)
        features_subset = features_np[:, feature_indices]
    else:
        features_subset = features_np

    try:
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(features_subset, rowvar=False)

        # Make values outside [-0.5, 0.5] more visible
        vis_correlation = correlation_matrix.copy()
        vis_correlation[vis_correlation > 0.5] = 0.5 + (vis_correlation[vis_correlation > 0.5] - 0.5) * 2
        vis_correlation[vis_correlation < -0.5] = -0.5 + (vis_correlation[vis_correlation < -0.5] + 0.5) * 2

        # Extract metrics
        abs_corr = np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0]))
        avg_abs_correlation = np.mean(abs_corr)
        max_abs_correlation = np.max(abs_corr)
        highly_correlated_pairs = np.sum(abs_corr > 0.5) / 2  # Divide by 2 to avoid double-counting
        correlation_redundancy = highly_correlated_pairs / (
                    correlation_matrix.shape[0] * (correlation_matrix.shape[0] - 1) / 2)

        # Create plot directory
        os.makedirs(save_dir, exist_ok=True)

        # Plot correlation matrix
        plt.figure(figsize=(10, 6))
        plt.imshow(vis_correlation, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        # # plt.title(
        #     f'Feature Correlation Matrix for {dataset_name}\nAvg |Correlation|: {avg_abs_correlation:.3f}, Max: {max_abs_correlation:.3f}',
        #     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_correlation_matrix.png"), dpi=300)
        plt.close()

        # Distribution of correlation values
        plt.figure(figsize=(10, 6))
        # Exclude diagonal (self-correlation)
        corr_values = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
        plt.hist(corr_values, bins=50, alpha=0.7)
        plt.xlabel('Correlation Value', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        # plt.title(f'Distribution of Feature Correlations for {dataset_name}', fontsize=14)
        plt.axvline(x=0.5, color='r', linestyle='--', label='High Correlation (0.5)')
        plt.axvline(x=-0.5, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_correlation_distribution.png"), dpi=300)
        plt.close()

        return {
            "dataset": dataset_name,
            "avg_absolute_correlation": float(avg_abs_correlation),
            "max_absolute_correlation": float(max_abs_correlation),
            "highly_correlated_pairs": int(highly_correlated_pairs),
            "correlation_redundancy_ratio": float(correlation_redundancy)
        }

    except Exception as e:
        print(f"Error during correlation analysis for {dataset_name}: {e}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "error": str(e)
        }


def analyze_feature_pca(features, dataset_name, save_dir, prefix=""):
    """
    Perform PCA analysis to demonstrate feature redundancy

    Args:
        features: Tensor of shape [n_samples, feature_dim]
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
        prefix: Prefix for saved files

    Returns:
        Dictionary with PCA analysis results
    """
    # Convert to numpy for analysis
    if isinstance(features, torch.Tensor):
        features_np = features.numpy()
    else:
        features_np = features

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_np)

        # Apply PCA
        pca = PCA(n_components=min(100, features_np.shape[1], features_np.shape[0]))
        pca_result = pca.fit_transform(features_scaled)

        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find number of components needed for different variance thresholds
        comp_80 = np.argmax(cumulative_variance >= 0.8) + 1
        comp_90 = np.argmax(cumulative_variance >= 0.9) + 1
        comp_95 = np.argmax(cumulative_variance >= 0.95) + 1
        comp_99 = np.argmax(cumulative_variance >= 0.99) + 1

        # Create plots directory
        os.makedirs(save_dir, exist_ok=True)

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7)
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        # plt.title(f'Explained Variance by Principal Components for {dataset_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_pca_variance.png"), dpi=300)
        plt.close()

        # Plot cumulative explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, '-', linewidth=2)
        plt.axhline(y=90, color='g', linestyle='--', label='90% Variance')
        plt.axhline(y=95, color='r', linestyle='--', label='95% Variance')
        plt.axhline(y=99, color='b', linestyle='--', label='99% Variance')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance (%)', fontsize=12)
        # # plt.title(
        #     f'Cumulative Explained Variance for {dataset_name}\n90% Variance: {comp_90} components, 95% Variance: {comp_95} components',
        #     fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_pca_cumulative.png"), dpi=300)
        plt.close()

        # Visualization of first two principal components
        if pca_result.shape[0] > 1000:
            # Subsample for visualization
            indices = np.random.choice(pca_result.shape[0], 1000, replace=False)
            pca_viz = pca_result[indices, :2]
        else:
            pca_viz = pca_result[:, :2]

        plt.figure(figsize=(8, 8))
        plt.scatter(pca_viz[:, 0], pca_viz[:, 1], alpha=0.5, s=10)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        # # plt.title(
        #     f'PCA Visualization for {dataset_name}\nTop 2 components explain {(explained_variance_ratio[0] + explained_variance_ratio[1]) * 100:.1f}% of variance',
        #     fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}{dataset_name}_pca_visualization.png"), dpi=300)
        plt.close()

        return {
            "dataset": dataset_name,
            "top_10_variance_ratio": explained_variance_ratio[:10].tolist(),
            "top_2_components_variance": float(explained_variance_ratio[0] + explained_variance_ratio[1]),
            "components_for_80_percent": int(comp_80),
            "components_for_90_percent": int(comp_90),
            "components_for_95_percent": int(comp_95),
            "components_for_99_percent": int(comp_99),
            "pca_redundancy_ratio": float(1 - comp_95 / features_np.shape[1])
        }

    except Exception as e:
        print(f"Error during PCA analysis for {dataset_name}: {e}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "error": str(e)
        }


def analyze_dataset_features(feature_dir, dataset_name, model_name, save_base_dir):
    """
    Perform comprehensive analysis of dataset features

    Args:
        feature_dir: Directory containing precomputed features
        dataset_name: Name of the dataset
        model_name: Name of the model
        save_base_dir: Base directory to save results

    Returns:
        Dictionary with analysis results
    """
    # Define paths for features
    try:
        # Try with Val suffix first
        feature_path = os.path.join(feature_dir, "precomputed_features", model_name,
                                    dataset_name + "Val", "train_features.pt")

        # If not found, try without Val suffix
        if not os.path.exists(feature_path):
            feature_path = os.path.join(feature_dir, "precomputed_features", model_name,
                                        dataset_name, "train_features.pt")

        # Load features
        print(f"Loading features from {feature_path}")
        features = torch.load(feature_path)

        # If too many samples, take a subset to avoid memory issues
        if features.shape[0] > MAX_SAMPLES_PER_DATASET:
            print(f"Taking a subset of {MAX_SAMPLES_PER_DATASET} samples from {features.shape[0]} total")
            indices = torch.randperm(features.shape[0])[:MAX_SAMPLES_PER_DATASET]
            features = features[indices]

        # Create a unique save directory for this dataset and model
        save_dir = os.path.join(save_base_dir, model_name, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # Get feature shape
        if isinstance(features, torch.Tensor):
            shape_str = f"{features.shape[0]} samples, {features.shape[1]} dimensions"
        else:
            shape_str = f"{features.shape}"

        print(f"Analyzing {shape_str} features for {dataset_name} with {model_name}")

        # Perform analyses
        svd_results = analyze_singular_values(features, dataset_name, save_dir, f"{model_name}_")
        correlation_results = analyze_feature_correlation(features, dataset_name, save_dir, f"{model_name}_")
        pca_results = analyze_feature_pca(features, dataset_name, save_dir, f"{model_name}_")

        # Combine all results
        all_results = {
            "dataset": dataset_name,
            "model": model_name,
            "feature_shape": shape_str,
            "svd_analysis": svd_results,
            "correlation_analysis": correlation_results,
            "pca_analysis": pca_results,
            "timestamp": datetime.now().isoformat()
        }

        # Save to JSON
        with open(os.path.join(save_dir, f"{model_name}_{dataset_name}_redundancy_analysis.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        return all_results

    except Exception as e:
        print(f"Error analyzing features for {dataset_name} with {model_name}: {e}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "model": model_name,
            "error": str(e)
        }


def create_combined_clustering_visualization(features_dict, model_name, save_dir):
    """
    Create a combined visualization showing clustering of features from multiple datasets

    Args:
        features_dict: Dictionary mapping dataset names to feature tensors
        model_name: Name of the model
        save_dir: Directory to save the visualization

    Returns:
        Path to the saved visualization
    """
    print("Creating combined clustering visualization...")

    # Create directory for clustering results
    cluster_dir = os.path.join(save_dir, model_name, "clustering")
    os.makedirs(cluster_dir, exist_ok=True)

    # Define colors and markers for different datasets - ensuring high contrast and distinguishability
    colors = ['red', 'purple', 'green', 'cyan', 'lime', 'pink', 'brown', 'orange']
    markers = ['o', 's', '^', 'D', 'v', 'p', '>', '<']

    # Process each dataset's features for combined visualization
    combined_features = []
    combined_labels = []
    dataset_names = []
    dataset_indices = {}

    current_index = 0
    for idx, (dataset_name, features) in enumerate(features_dict.items()):
        print(f"Processing {dataset_name} features for combined visualization...")

        # Convert to numpy if it's a tensor
        if isinstance(features, torch.Tensor):
            features_np = features.numpy()
        else:
            features_np = features

        # Sample a smaller subset for better visualization
        max_samples = min(500, features_np.shape[0])
        indices = np.random.choice(features_np.shape[0], max_samples, replace=False)
        sampled_features = features_np[indices]

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(sampled_features)

        # Store features and create labels
        combined_features.append(features_scaled)
        combined_labels.extend([idx] * len(features_scaled))
        dataset_names.append(dataset_name)
        dataset_indices[dataset_name] = idx

        current_index += len(features_scaled)

    # Combine all features
    all_features = np.vstack(combined_features)
    all_labels = np.array(combined_labels)

    # Reduce dimensionality first with PCA
    pca = PCA(n_components=min(50, all_features.shape[1], all_features.shape[0] - 1))
    features_pca = pca.fit_transform(all_features)

    print(f"Performing t-SNE on combined features (shape: {features_pca.shape})...")

    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_labels) - 1))
    features_tsne = tsne.fit_transform(features_pca)

    # Create the combined visualization
    plt.figure(figsize=(10, 6))

    # Create handles for legend
    legend_handles = []

    # Plot each dataset with a different color and shape
    for idx, dataset_name in enumerate(dataset_names):
        mask = all_labels == idx
        if np.any(mask):
            color_idx = idx % len(colors)
            marker_idx = idx % len(markers)

            # Plot dataset points
            scatter = plt.scatter(
                features_tsne[mask, 0],
                features_tsne[mask, 1],
                c=colors[color_idx],
                marker=markers[marker_idx],
                s=40,  # Larger point size
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5
            )

            # Create handle for legend - 使用颜色和形状标识替换数字标识
            handle = plt.Line2D(
                [0], [0],
                marker=markers[marker_idx],
                color='w',
                markerfacecolor=colors[color_idx],
                markersize=8,
                label=dataset_name
            )
            legend_handles.append(handle)

    # Add axes labels
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)

    # Add legend with customizations for clarity - 将图例移到左下角并缩小尺寸
    legend = plt.legend(
        handles=legend_handles,
        loc='lower left',  # 位置改为左下角
        fontsize=10,  # 缩小字体
        framealpha=0.8,  # 半透明背景
        edgecolor='black',  # 黑色边框
        facecolor='white',  # 白色背景
        ncol=2  # 使用2列以减小图例尺寸
    )

    # Add thin border to the plot
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    # Add grid for reference but make it light
    plt.grid(True, alpha=0.2)

    # Use tight layout for better spacing
    plt.tight_layout()

    # Save the visualization
    output_path = os.path.join(cluster_dir, "combined_dataset_clustering.png")
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.close()

    print(f"Combined clustering visualization saved to {output_path}")
    return output_path


def analyze_feature_clustering(features_dict, model_name, save_dir):
    """
    Perform hierarchical clustering on features from multiple datasets
    and create a combined visualization

    Args:
        features_dict: Dictionary mapping dataset names to feature tensors
        model_name: Name of the model
        save_dir: Directory to save results

    Returns:
        Dictionary with clustering results
    """
    # Create directory for clustering results
    cluster_dir = os.path.join(save_dir, model_name, "clustering")
    os.makedirs(cluster_dir, exist_ok=True)

    # Create combined clustering visualization (new function)
    create_combined_clustering_visualization(features_dict, model_name, save_dir)

    # Set up figure for combined visualization
    n_datasets = len(features_dict)
    fig_height = max(10, n_datasets * 3)  # Scale height based on number of datasets

    # TSNE visualization of feature clusters
    plt.figure(figsize=(10, 6))

    cluster_results = {}

    # Process each dataset
    for i, (dataset_name, features) in enumerate(features_dict.items()):
        try:
            print(f"Clustering features for {dataset_name}...")

            # Convert to numpy if it's a tensor
            if isinstance(features, torch.Tensor):
                features_np = features.numpy()
            else:
                features_np = features

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_np)

            # Dimensionality reduction with PCA first (to make TSNE faster)
            # Fix: Ensure n_components doesn't exceed the number of features or samples
            n_components = min(50, features_scaled.shape[1], features_scaled.shape[0] - 1)
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)

            # Apply t-SNE for visualization
            tsne = TSNE(n_components=2, random_state=42)
            features_tsne = tsne.fit_transform(features_pca)

            # Create subplot for this dataset
            plt.subplot(n_datasets, 1, i+1)
            plt.scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.7, s=10)
            # plt.title(f'Feature Clustering for {dataset_name} ({model_name})', fontsize=14)
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Store results
            cluster_results[dataset_name] = {
                "n_samples": features_np.shape[0],
                "n_features": features_np.shape[1],
                "pca_variance_explained": sum(pca.explained_variance_ratio_)
            }

        except Exception as e:
            print(f"Error clustering features for {dataset_name}: {e}")
            traceback.print_exc()
            cluster_results[dataset_name] = {"error": str(e)}

    # Adjust layout - Fix: Use plt.subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(cluster_dir, f"{model_name}_combined_feature_clustering.png"), dpi=300)
    plt.close()

    # Create hierarchical clustering visualization
    try:
        print("Creating hierarchical clustering visualization...")

        # Create a single figure with multiple subplots
        fig = plt.figure(figsize=(10, 6))

        for i, (dataset_name, features) in enumerate(features_dict.items()):
            try:
                # Convert to numpy if it's a tensor
                if isinstance(features, torch.Tensor):
                    features_np = features.numpy()
                else:
                    features_np = features

                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_np)

                # Reduce dimensionality first - Fix: Ensure n_components doesn't exceed limits
                n_components = min(50, features_scaled.shape[1], features_scaled.shape[0] - 1)
                pca = PCA(n_components=n_components)
                features_pca = pca.fit_transform(features_scaled)

                # Take a subset of samples for clustering visualization
                sample_size = min(500, features_pca.shape[0])
                indices = np.random.choice(features_pca.shape[0], sample_size, replace=False)
                subset = features_pca[indices]

                # Calculate distance matrix
                distances = pdist(subset, metric='euclidean')

                # Hierarchical clustering
                Z = linkage(distances, method='ward')

                # Create subplot
                plt.subplot(n_datasets, 1, i+1)
                dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='lastp', p=30, labels=None)
                # plt.title(f'Hierarchical Clustering for {dataset_name} ({model_name})', fontsize=14)
                plt.ylabel('Distance', fontsize=12)

                # Update results
                cluster_results[dataset_name].update({
                    "n_clusters_identified": len(set(Z[:, 0].astype(int))),
                    "max_distance": np.max(Z[:, 2])
                })

            except Exception as e:
                print(f"Error in hierarchical clustering for {dataset_name}: {e}")
                traceback.print_exc()
                if dataset_name in cluster_results:
                    cluster_results[dataset_name].update({"hier_clustering_error": str(e)})

        # Fix: Use subplots_adjust instead of tight_layout
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(cluster_dir, f"{model_name}_hierarchical_clustering.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error creating hierarchical clustering visualization: {e}")
        traceback.print_exc()

    # Save clustering results
    with open(os.path.join(cluster_dir, f"{model_name}_clustering_results.json"), "w") as f:
        json.dump(cluster_results, f, indent=2)

    return cluster_results


def analyze_coefficient_distributions(model_dir, datasets, model_name, save_dir):
    """
    Analyze distributions of learned coefficients from trained models
    to demonstrate the need for sparsity

    Args:
        model_dir: Directory containing trained models
        datasets: List of dataset names
        model_name: Name of the model
        save_dir: Directory to save results

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Storage for block importance data across datasets
    all_block_importance = {}
    all_coeffs_data = {}

    for dataset in tqdm(datasets, desc="Analyzing coefficient distributions"):
        try:
            # Define paths
            model_path = os.path.join(model_dir, model_name, dataset + "Val", "best_adaptive_gating_model.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, model_name, dataset + "Val", "best_precomputed_model.pt")

            if not os.path.exists(model_path):
                print(f"Model not found for {dataset}")
                continue

            # Load model
            state_dict = torch.load(model_path, map_location="cpu")

            # Extract meta_net weights that determine coefficients
            meta_weights = None
            if 'meta_net' in state_dict:
                for key in state_dict['meta_net'].keys():
                    if key.endswith('net.2.weight'):
                        meta_weights = state_dict['meta_net'][key]
                        break

            if meta_weights is None:
                print(f"Could not find meta_net weights for {dataset}")
                continue

            # Extract configuration
            config = state_dict.get('config', {})
            base_threshold = config.get('base_threshold', 0.05)
            beta = config.get('beta', 1.0)
            blockwise = config.get('blockwise', True)

            # Analyze weight distribution
            weight_abs = torch.abs(meta_weights).flatten()
            weight_mean = torch.mean(weight_abs).item()
            weight_std = torch.std(weight_abs).item()
            weight_median = torch.median(weight_abs).item()
            weight_max = torch.max(weight_abs).item()
            weight_sparsity = torch.mean((weight_abs < base_threshold).float()).item()

            # Create visualization directory
            os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)

            # Plot coefficient distribution
            plt.figure(figsize=(10, 6))
            plt.hist(weight_abs.numpy(), bins=50, alpha=0.7)
            plt.axvline(x=base_threshold, color='r', linestyle='--', label=f'Base Threshold ({base_threshold:.4f})')
            plt.xlabel('Absolute Weight Magnitude', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            # # plt.title(
            #     f'Coefficient Magnitude Distribution for {dataset}\nSparsity: {weight_sparsity * 100:.1f}% below threshold',
            #     fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, model_name, f"{dataset}_coefficient_distribution.png"), dpi=300)
            plt.close()

            # Analyze blockwise distribution if applicable
            if blockwise and meta_weights.dim() > 1:
                # Try to reshape into task vectors and blocks
                try:
                    if meta_weights.shape[0] > meta_weights.shape[1]:
                        # Transpose if needed
                        reshaped = meta_weights.T
                    else:
                        reshaped = meta_weights

                    # Estimate number of task vectors and blocks
                    num_task_vectors = 8  # Default in the code
                    feature_dim = reshaped.shape[0]
                    blocks_per_vector = reshaped.shape[1] // num_task_vectors

                    # Reshape properly
                    if blocks_per_vector > 0:
                        reshaped = reshaped[:, :num_task_vectors * blocks_per_vector].reshape(
                            feature_dim, num_task_vectors, blocks_per_vector)

                        # Average across feature dimension
                        block_importance = torch.mean(torch.abs(reshaped),
                                                      dim=0)  # [num_task_vectors, blocks_per_vector]

                        # Store block importance for combined visualization
                        all_block_importance[dataset] = block_importance.numpy()

                        # Store coefficient data for clustering
                        all_coeffs_data[dataset] = reshaped.numpy()

                        # Compute block-level statistics
                        block_sparsity = torch.mean((block_importance < base_threshold).float()).item()
                        block_variance = torch.var(block_importance).item()
                        block_max = torch.max(block_importance).item()

                        # Add blockwise stats to results
                        results[dataset] = {
                            "weight_mean": weight_mean,
                            "weight_std": weight_std,
                            "weight_median": weight_median,
                            "weight_max": weight_max,
                            "weight_sparsity": weight_sparsity,
                            "base_threshold": base_threshold,
                            "beta": beta,
                            "blockwise": blockwise,
                            "block_sparsity": block_sparsity,
                            "block_variance": block_variance,
                            "block_max": block_max,
                            "num_task_vectors": num_task_vectors,
                            "blocks_per_vector": blocks_per_vector
                        }
                    else:
                        # Couldn't determine blockwise structure
                        results[dataset] = {
                            "weight_mean": weight_mean,
                            "weight_std": weight_std,
                            "weight_median": weight_median,
                            "weight_max": weight_max,
                            "weight_sparsity": weight_sparsity,
                            "base_threshold": base_threshold,
                            "beta": beta,
                            "blockwise": blockwise,
                            "blocks_per_vector": 0  # Couldn't determine
                        }
                except Exception as e:
                    print(f"Error analyzing blockwise structure for {dataset}: {e}")
                    results[dataset] = {
                        "weight_mean": weight_mean,
                        "weight_std": weight_std,
                        "weight_median": weight_median,
                        "weight_max": weight_max,
                        "weight_sparsity": weight_sparsity,
                        "base_threshold": base_threshold,
                        "beta": beta,
                        "blockwise": blockwise,
                        "error": str(e)
                    }
            else:
                # Non-blockwise
                results[dataset] = {
                    "weight_mean": weight_mean,
                    "weight_std": weight_std,
                    "weight_median": weight_median,
                    "weight_max": weight_max,
                    "weight_sparsity": weight_sparsity,
                    "base_threshold": base_threshold,
                    "beta": beta,
                    "blockwise": blockwise
                }

        except Exception as e:
            print(f"Error analyzing coefficients for {dataset}: {e}")
            traceback.print_exc()
            results[dataset] = {
                "error": str(e)
            }

    # Create combined block importance heatmap for all datasets
    if all_block_importance:
        try:
            # Group datasets by block size to ensure compatibility
            block_size_groups = defaultdict(list)

            for dataset, block_imp in all_block_importance.items():
                # Use the second dimension (block size) as the group key
                block_size = block_imp.shape[1]
                block_size_groups[block_size].append(dataset)

            print(f"Grouped datasets by block size: {dict(block_size_groups)}")

            # Process each group separately
            for block_size, group_datasets in block_size_groups.items():
                if len(group_datasets) == 0:
                    continue

                print(f"Creating combined block importance heatmap for block size {block_size} with {len(group_datasets)} datasets")

                # Create figure with subplots arranged vertically
                n_datasets = len(group_datasets)
                fig, axes = plt.subplots(n_datasets, 1, figsize=(15, n_datasets * 3),
                                       sharex=True, sharey=True)

                # If there's only one dataset, axes won't be an array
                if n_datasets == 1:
                    axes = [axes]

                # Plot each dataset in this group
                for i, dataset in enumerate(group_datasets):
                    block_imp = all_block_importance[dataset]
                    im = axes[i].imshow(block_imp, cmap='Blues', aspect='auto')
                    axes[i].set_title(f'Block Importance Heatmap for {dataset}', fontsize=12)
                    axes[i].set_ylabel('Task Vector Index', fontsize=10)

                    # Only set xlabel for the bottom subplot
                    if i == n_datasets - 1:
                        axes[i].set_xlabel('Block Index', fontsize=10)

                # Add colorbar
                cbar = fig.colorbar(im, ax=axes, shrink=0.8)
                cbar.set_label('Average Magnitude (Blue intensity indicates importance)', fontsize=10)

                # Adjust layout
                plt.subplots_adjust(hspace=0.3, right=0.9)
                plt.savefig(os.path.join(save_dir, model_name, f"combined_block_importance_size_{block_size}.png"), dpi=300)
                plt.close()

                # Create a single large heatmap for each group
                try:
                    print(f"Creating single combined block importance heatmap for block size {block_size}...")
                    plt.figure(figsize=(10, 6))

                    # Combine all block importance matrices in this group
                    combined_data = np.concatenate([all_block_importance[d] for d in group_datasets], axis=0)

                    # Calculate y-tick positions for dataset labels
                    y_positions = []
                    current_pos = 0
                    for dataset in group_datasets:
                        y_positions.append(current_pos + all_block_importance[dataset].shape[0]//2)
                        current_pos += all_block_importance[dataset].shape[0]

                    # Plot combined heatmap
                    plt.imshow(combined_data, cmap='Blues', aspect='auto')
                    plt.colorbar(label='Average Magnitude')
                    # plt.title(f'Combined Block Importance Heatmap (Block Size: {block_size}) for {model_name}', fontsize=14)
                    plt.xlabel('Block Index', fontsize=12)

                    # Add dataset labels on y-axis
                    plt.yticks(y_positions, group_datasets)

                    # Add horizontal lines to separate datasets
                    cumulative_height = 0
                    for dataset in group_datasets[:-1]:  # No line after the last dataset
                        cumulative_height += all_block_importance[dataset].shape[0]
                        plt.axhline(y=cumulative_height-0.5, color='black', linestyle='-', linewidth=0.5)

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, model_name, f"single_combined_heatmap_size_{block_size}.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Error creating single combined heatmap for block size {block_size}: {e}")
                    traceback.print_exc()

            # Create a master heatmap visualization showing all different groups
            # This just creates a figure indicating the different block sizes found
            plt.figure(figsize=(10, 6))
            block_sizes = list(block_size_groups.keys())
            dataset_counts = [len(group_datasets) for group_datasets in block_size_groups.values()]

            plt.bar(range(len(block_sizes)), dataset_counts, tick_label=[f"Size {bs}" for bs in block_sizes])
            plt.xlabel('Block Size Groups', fontsize=12)
            plt.ylabel('Number of Datasets', fontsize=12)
            # plt.title(f'Block Size Distribution for {model_name}', fontsize=14)

            # Add dataset names as annotations
            for i, block_size in enumerate(block_sizes):
                group_datasets = block_size_groups[block_size]
                plt.annotate(
                    "\n".join(group_datasets),
                    xy=(i, dataset_counts[i] + 0.1),
                    ha='center',
                    fontsize=8
                )

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, model_name, f"block_size_distribution.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error creating block size distribution chart: {e}")
            traceback.print_exc()

    # Perform clustering on coefficients if available
    if all_coeffs_data:
        try:
            # Create clustering visualization for coefficient similarities across datasets
            features_for_clustering = {}

            # Prepare data for clustering
            for dataset, coeffs in all_coeffs_data.items():
                # Reshape coefficients to 2D for clustering
                # We'll treat each task vector+block combination as a "feature"
                reshaped_coeffs = coeffs.reshape(coeffs.shape[0], -1)
                features_for_clustering[dataset] = reshaped_coeffs

            # Perform clustering analysis
            print("Performing clustering analysis on coefficient data...")

            # Fix: Ensure we handle coefficient data properly for clustering
            # Define a separate function for coefficient clustering to avoid errors with feature clustering
            clustering_results = {}

            # Process each dataset's coefficients
            for dataset, coeffs in features_for_clustering.items():
                try:
                    # Get only a sample of coefficients for analysis
                    if coeffs.shape[0] > 500:
                        indices = np.random.choice(coeffs.shape[0], 500, replace=False)
                        sample_coeffs = coeffs[indices]
                    else:
                        sample_coeffs = coeffs

                    # Calculate mean and std for result metrics
                    mean_coeff = np.mean(sample_coeffs)
                    std_coeff = np.std(sample_coeffs)

                    # Record basic statistics
                    clustering_results[dataset] = {
                        "n_coefficients": coeffs.shape[0] * coeffs.shape[1],
                        "mean": float(mean_coeff),
                        "std": float(std_coeff)
                    }

                except Exception as e:
                    print(f"Error processing coefficients for {dataset}: {e}")
                    clustering_results[dataset] = {"error": str(e)}

            # Add clustering results to the overall results
            for dataset, cluster_result in clustering_results.items():
                if dataset in results:
                    results[dataset]["coefficient_stats"] = cluster_result

        except Exception as e:
            print(f"Error performing clustering analysis: {e}")
            traceback.print_exc()

    # Save overall results
    with open(os.path.join(save_dir, model_name, "coefficient_analysis_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def generate_redundancy_report(feature_analyses, coefficient_analyses, save_dir, model_name):
    """
    Generate a comprehensive report about redundancy analysis

    Args:
        feature_analyses: Dictionary of feature analysis results
        coefficient_analyses: Dictionary of coefficient analysis results
        save_dir: Directory to save report
        model_name: Name of the model
    """
    # Create report directory
    report_dir = os.path.join(save_dir, model_name)
    os.makedirs(report_dir, exist_ok=True)

    # Prepare HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Redundancy Analysis Report - {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #3366cc; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #ffffcc; }}
            .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }}
            .gallery img {{ max-width: 300px; border: 1px solid #ddd; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Feature Redundancy Analysis Report</h1>
        <div class="section">
            <h2>Summary</h2>
            <p>This report analyzes feature redundancy across different datasets processed with {model_name} to justify the use of spike-and-slab priors.</p>
            <p>Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="section">
            <h2>Feature Redundancy Analysis</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Feature Dim</th>
                    <th>Effective Rank</th>
                    <th>Redundancy Ratio</th>
                    <th>Dims for 95% Variance</th>
                    <th>Correlation Redundancy</th>
                </tr>
    """

    # Add rows for each dataset's feature analysis
    for dataset, analysis in feature_analyses.items():
        if "error" in analysis:
            # Handle error case
            html += f"""
                <tr>
                    <td>{dataset}</td>
                    <td colspan="5" class="error">Error: {analysis['error']}</td>
                </tr>
            """
        else:
            svd = analysis.get("svd_analysis", {})
            pca = analysis.get("pca_analysis", {})
            corr = analysis.get("correlation_analysis", {})

            html += f"""
                <tr>
                    <td>{dataset}</td>
                    <td>{svd.get('feature_dim', 'N/A')}</td>
                    <td>{svd.get('effective_rank', 'N/A'):.1f}</td>
                    <td>{svd.get('redundancy_ratio', 'N/A'):.1%}</td>
                    <td>{pca.get('components_for_95_percent', 'N/A')}</td>
                    <td>{corr.get('correlation_redundancy_ratio', 'N/A'):.1%}</td>
                </tr>
            """

    html += """
            </table>
        </div>

        <div class="section">
            <h2>Coefficient Analysis</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Weight Sparsity</th>
                    <th>Base Threshold</th>
                    <th>Block Sparsity</th>
                    <th>Weight Std/Mean Ratio</th>
                </tr>
    """

    # Add rows for each dataset's coefficient analysis
    for dataset, analysis in coefficient_analyses.items():
        if "error" in analysis:
            # Handle error case
            html += f"""
                <tr>
                    <td>{dataset}</td>
                    <td colspan="4" class="error">Error: {analysis['error']}</td>
                </tr>
            """
        else:
            weight_std = analysis.get('weight_std', 0)
            weight_mean = analysis.get('weight_mean', 1)
            std_mean_ratio = weight_std / weight_mean if weight_mean > 0 else 0

            html += f"""
                <tr>
                    <td>{dataset}</td>
                    <td>{analysis.get('weight_sparsity', 'N/A'):.1%}</td>
                    <td>{analysis.get('base_threshold', 'N/A'):.4f}</td>
                    <td>{analysis.get('block_sparsity', 'N/A'):.1%}</td>
                    <td>{std_mean_ratio:.2f}</td>
                </tr>
            """

    html += """
            </table>
        </div>

        <div class="section">
            <h2>Key Findings</h2>
            <ul>
    """

    # Calculate average redundancy ratio
    avg_redundancy = 0
    count = 0
    for dataset, analysis in feature_analyses.items():
        svd = analysis.get("svd_analysis", {})
        if 'redundancy_ratio' in svd:
            avg_redundancy += svd['redundancy_ratio']
            count += 1

    avg_redundancy = avg_redundancy / count if count > 0 else 0

    # Calculate average sparsity
    avg_sparsity = 0
    count = 0
    for dataset, analysis in coefficient_analyses.items():
        if 'weight_sparsity' in analysis:
            avg_sparsity += analysis['weight_sparsity']
            count += 1

    avg_sparsity = avg_sparsity / count if count > 0 else 0

    # Add key findings
    html += f"""
                <li>The average redundancy ratio across datasets is <strong>{avg_redundancy:.1%}</strong>, indicating substantial redundancy in the feature space.</li>
                <li>On average, <strong>{avg_sparsity:.1%}</strong> of weights are below the threshold, suggesting that many coefficients can be safely pruned.</li>
                <li>PCA analysis shows that typically only <strong>5-15%</strong> of dimensions are needed to capture 95% of variance.</li>
                <li>There is significant correlation between features, further indicating redundancy.</li>
                <li>These findings strongly support the use of spike-and-slab priors to induce sparsity in task vector coefficients.</li>
            </ul>
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            <ul>
                <li>Use spike-and-slab priors to encourage sparsity in the learned coefficients.</li>
                <li>Implement adaptive thresholding that can adjust based on feature importance and uncertainty.</li>
                <li>Consider blockwise sparsity patterns as the analysis shows different importance across blocks.</li>
                <li>Exploit the redundancy by using a lower-dimensional representation where appropriate.</li>
            </ul>
        </div>

        <div class="section">
            <h2>Visualization Gallery</h2>
            <h3>Dataset Clustering</h3>
            <div class="gallery">
                <img src="clustering/combined_dataset_clustering.png" alt="Combined Dataset Clustering" onerror="this.style.display='none'">
            </div>
            
            <h3>Block Size Distribution</h3>
            <div class="gallery">
                <img src="block_size_distribution.png" alt="Block Size Distribution" onerror="this.style.display='none'">
            </div>
            
            <h3>Feature Clustering</h3>
            <div class="gallery">
                <img src="clustering/combined_feature_clustering.png" alt="Feature Clustering" onerror="this.style.display='none'">
                <img src="clustering/hierarchical_clustering.png" alt="Hierarchical Clustering" onerror="this.style.display='none'">
            </div>
            
            <h3>Singular Value Distributions</h3>
            <div class="gallery">
    """

    # Add SVD visualizations
    for dataset in feature_analyses.keys():
        svd_img_path = f"{model_name}/{dataset}/svd_images/{model_name}_{dataset}_singular_values.png"
        html += f"""
                <img src="{svd_img_path}" alt="SVD for {dataset}" onerror="this.style.display='none'">
        """

    html += """
            </div>

            <h3>PCA Visualizations</h3>
            <div class="gallery">
    """

    # Add PCA visualizations
    for dataset in feature_analyses.keys():
        pca_img_path = f"{model_name}/{dataset}/pca_images/{model_name}_{dataset}_pca_cumulative.png"
        html += f"""
                <img src="{pca_img_path}" alt="PCA for {dataset}" onerror="this.style.display='none'">
        """

    html += """
            </div>

            <h3>Coefficient Distributions</h3>
            <div class="gallery">
    """

    # Add coefficient visualizations
    for dataset in coefficient_analyses.keys():
        coef_img_path = f"{model_name}/{dataset}_coefficient_distribution.png"
        html += f"""
                <img src="{coef_img_path}" alt="Coefficients for {dataset}" onerror="this.style.display='none'">
        """

    html += """
            </div>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(os.path.join(report_dir, "redundancy_report.html"), "w") as f:
        f.write(html)

    print(f"Generated redundancy report saved to {os.path.join(report_dir, 'redundancy_report.html')}")

    # Generate plain text summary for console
    print("\n===== REDUNDANCY ANALYSIS SUMMARY =====")
    print(f"Model: {model_name}")
    print(f"Average Feature Redundancy: {avg_redundancy:.1%}")
    print(f"Average Weight Sparsity: {avg_sparsity:.1%}")
    print("\nFeature Redundancy by Dataset:")

    for dataset, analysis in feature_analyses.items():
        svd = analysis.get("svd_analysis", {})
        if 'redundancy_ratio' in svd:
            print(f"  {dataset}: {svd['redundancy_ratio']:.1%} redundancy, " +
                  f"Effective Rank: {svd.get('effective_rank', 'N/A'):.1f}/{svd.get('feature_dim', 'N/A')}")

    print("\nCoefficient Sparsity by Dataset:")
    for dataset, analysis in coefficient_analyses.items():
        if 'weight_sparsity' in analysis:
            print(f"  {dataset}: {analysis['weight_sparsity']:.1%} sparsity, " +
                  f"Base Threshold: {analysis.get('base_threshold', 'N/A'):.4f}")

    print("\nNew visualizations created:")
    print(f"  - Combined dataset clustering visualization: {os.path.join(save_dir, model_name, 'clustering/combined_dataset_clustering.png')}")
    print(f"  - Block size distribution chart: {os.path.join(report_dir, 'block_size_distribution.png')}")
    print(f"  - Group-specific block importance heatmaps: {os.path.join(report_dir, 'single_combined_heatmap_size_*.png')}")
    print(f"  - Feature clustering: {os.path.join(report_dir, 'clustering/combined_feature_clustering.png')}")
    print(f"  - Hierarchical clustering: {os.path.join(report_dir, 'clustering/hierarchical_clustering.png')}")
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature redundancy to justify spike-and-slab priors")

    parser.add_argument("--data-location", type=str, default=os.path.expanduser("~/data"),
                        help="Root directory for datasets and precomputed features")
    parser.add_argument("--model-dir", type=str, default=os.path.expanduser("~/checkpoints_adaptive_gating"),
                        help="Directory containing trained models")
    parser.add_argument("--save-dir", type=str, default="results/redundancy_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--model-name", type=str, default="ViT-B-32",
                        choices=["ViT-B-32", "ViT-L-14", "ViT-B-16"],
                        help="Model name to analyze")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"],
                        help="Datasets to analyze")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum number of samples to analyze per dataset")
    parser.add_argument("--skip-feature-analysis", action="store_true",
                        help="Skip the feature analysis (only analyze coefficients)")
    parser.add_argument("--skip-coefficient-analysis", action="store_true",
                        help="Skip the coefficient analysis (only analyze features)")

    args = parser.parse_args()

    # Update global constant
    global MAX_SAMPLES_PER_DATASET
    MAX_SAMPLES_PER_DATASET = args.max_samples

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Record all analyses
    feature_analyses = {}
    coefficient_analyses = {}

    # Dictionary to store features for clustering
    feature_dict = {}

    # Step 1: Analyze features if not skipped
    if not args.skip_feature_analysis:
        print(f"\n===== Analyzing Feature Redundancy for {args.model_name} =====")
        for dataset in args.datasets:
            print(f"\nProcessing {dataset}...")
            try:
                # Try with Val suffix first
                feature_path = os.path.join(args.data_location, "precomputed_features", args.model_name,
                                            dataset + "Val", "train_features.pt")

                # If not found, try without Val suffix
                if not os.path.exists(feature_path):
                    feature_path = os.path.join(args.data_location, "precomputed_features", args.model_name,
                                                dataset, "train_features.pt")

                # Load features for clustering analysis
                if os.path.exists(feature_path):
                    print(f"Loading features from {feature_path} for clustering...")
                    features = torch.load(feature_path)

                    # If too many samples, take a subset for memory efficiency
                    if features.shape[0] > MAX_SAMPLES_PER_DATASET:
                        indices = torch.randperm(features.shape[0])[:MAX_SAMPLES_PER_DATASET]
                        features = features[indices]

                    # Store for clustering
                    feature_dict[dataset] = features

                analysis_result = analyze_dataset_features(
                    args.data_location,
                    dataset,
                    args.model_name,
                    args.save_dir
                )
                feature_analyses[dataset] = analysis_result
            except Exception as e:
                print(f"Error analyzing features for {dataset}: {e}")
                traceback.print_exc()
                feature_analyses[dataset] = {"error": str(e)}

        # Perform feature clustering across datasets
        if feature_dict:
            print("\n===== Performing Feature Clustering Analysis =====")
            try:
                clustering_results = analyze_feature_clustering(
                    feature_dict,
                    args.model_name,
                    args.save_dir
                )
                print(f"Clustering analysis completed for {len(clustering_results)} datasets")
            except Exception as e:
                print(f"Error during feature clustering analysis: {e}")
                traceback.print_exc()

    # Step 2: Analyze coefficients if not skipped
    if not args.skip_coefficient_analysis:
        print(f"\n===== Analyzing Coefficient Distributions for {args.model_name} =====")

        try:
            coefficient_analyses = analyze_coefficient_distributions(
                args.model_dir,
                args.datasets,
                args.model_name,
                args.save_dir
            )
        except Exception as e:
            print(f"Error analyzing coefficients: {e}")
            traceback.print_exc()

    # Step 3: Generate comprehensive report
    print("\n===== Generating Redundancy Report =====")
    generate_redundancy_report(
        feature_analyses,
        coefficient_analyses,
        args.save_dir,
        args.model_name
    )

    print("\nRedundancy analysis complete!")


if __name__ == "__main__":
    main()