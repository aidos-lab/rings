"""
Complementarity Analysis for Graph Datasets using RINGS Framework

This script demonstrates how to:
1. Load a graph dataset from PyTorch Geometric
2. Apply various RINGS perturbations to the graph data
3. Compute complementarity metrics

Usage (from root directory):
    python -m examples.complementarity --dataset MUTAG --perturbation original

For more options:
    python -m examples.complementarity --help
"""

import numpy as np
import torch
import argparse
import time
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from rings.complementarity import ComplementarityFunctor, MatrixNormComparator
from rings.perturbations import (
    CompleteGraph,
    EmptyFeatures,
    EmptyGraph,
    Original,
    RandomFeatures,
    RandomGraph,
)


def get_available_perturbations():
    """
    Return a list of available perturbation names with descriptions.

    Returns:
        dict: Mapping from perturbation name to description
    """
    return {
        "original": "No perturbation (original graph and features)",
        # Node feature perturbations
        "empty-features": "Replace node features with empty (zero) vectors",
        "random-features": "Replace node features with random values",
        # Graph structure perturbations
        "empty-graph": "Remove all edges from the graph",
        "complete-graph": "Create edges between all pairs of nodes",
        "random-graph": "Generate a random graph structure",
    }


def create_perturbation(name, seed=42):
    """
    Create a perturbation transform based on the given name and seed.

    Args:
        name (str): Name of the perturbation
        seed (int): Random seed for reproducible perturbations

    Returns:
        BaseTransform: A transform that can be applied to graph data
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    if name == "original":
        return Original()

    # Node feature perturbations
    elif name == "empty-features":
        return EmptyFeatures()
    elif name == "random-features":
        return RandomFeatures(fixed_dimension=None)

    # Graph structure perturbations
    elif name == "empty-graph":
        return EmptyGraph()
    elif name == "complete-graph":
        return CompleteGraph()
    elif name == "random-graph":
        return RandomGraph(shuffle=True)

    else:
        raise ValueError(f"Unknown perturbation: {name}")


def apply_perturbation(dataset, perturbation):
    """
    Apply a perturbation to all graphs in a dataset.

    Args:
        dataset: PyTorch Geometric dataset
        perturbation: Transform to apply

    Returns:
        list: List of transformed graph data objects
    """
    print(f"Applying perturbation: {perturbation.__class__.__name__}")
    transformed_dataset = [perturbation(data.clone()) for data in dataset]
    return transformed_dataset


def create_complementarity_functor(n_jobs=1):
    """
    Create a ComplementarityFunctor with standard parameters.

    The ComplementarityFunctor measures the difference between:
    1. The metric space of node features (using euclidean distance)
    2. The metric space of graph structure (using shortest path distance)

    Args:
        n_jobs (int): Number of parallel jobs (-1 for all cores)

    Returns:
        ComplementarityFunctor: Configured functor
    """
    return ComplementarityFunctor(
        # Metric for node features (pairwise Euclidean distance)
        feature_metric="euclidean",
        # Metric for graph structure (shortest path distance between nodes)
        graph_metric="shortest_path_distance",
        # Method to compare the two metric spaces
        comparator=MatrixNormComparator,
        # Parallelization parameter
        n_jobs=n_jobs,
        # Norm used by the comparator
        norm="L11",
    )


def compute_complementarity(dataloader, functor):
    """
    Compute complementarity scores for all graphs in the dataloader.

    Args:
        dataloader: PyTorch Geometric dataloader containing graphs
        functor: ComplementarityFunctor to compute scores

    Returns:
        numpy.ndarray: Array of complementarity scores
    """
    start_time = time.time()
    print("Computing complementarity scores...")

    all_scores = []
    for batch in dataloader:
        # The functor computes complementarity for each graph in the batch
        results = functor(batch)

        # Extract the complementarity scores from the results
        batch_scores = results["complementarity"]
        all_scores.extend(batch_scores.tolist())

    duration = time.time() - start_time
    print(f"Computation completed in {duration:.2f} seconds")

    return np.array(all_scores)


def main():
    """Main function to run the complementarity analysis.

    User inputs are handled via command line arguments:
        --perturbation: "Perturbation to apply to the dataset (default: original)"
        --dataset: "Name of the TU dataset to use (default: MUTAG)"
        --seed: "Random seed for reproducibility (default: 42)"
        --batch-size: "Batch size for dataloader (default: 32)"
        --n-jobs: "Number of parallel jobs (-1 for all cores, default: 1)"
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute complementarity metrics for graph datasets"
    )

    # Get available perturbations for help text
    perturbations = get_available_perturbations()
    perturbation_help = "Available perturbations:\n" + "\n".join(
        f"  {name}: {desc}" for name, desc in perturbations.items()
    )

    parser.add_argument(
        "--perturbation",
        type=str,
        default="original",
        choices=list(perturbations.keys()),
        help=f"Perturbation to apply to the dataset. {perturbation_help}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MUTAG",
        help="Name of the TU dataset to use (e.g., MUTAG, ENZYMES, PROTEINS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dataloader",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (-1 for all cores)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = TUDataset(root="/tmp/TUDataset", name=args.dataset)
    print(f"Dataset loaded: {len(dataset)} graphs")

    # 2. Create perturbation and apply to dataset
    perturbation = create_perturbation(args.perturbation, args.seed)
    transformed_dataset = apply_perturbation(dataset, perturbation)

    # 3. Create dataloader for batch processing
    dataloader = DataLoader(
        transformed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 4. Create complementarity functor
    functor = create_complementarity_functor(n_jobs=args.n_jobs)

    # 5. Compute complementarity scores
    scores = compute_complementarity(dataloader, functor)

    # 6. Calculate and print statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Print results
    print("\n" + "=" * 60)
    print("📊 Results Summary".center(60))
    print("=" * 60)
    print(f"📁 Dataset:         {args.dataset}")
    print(f"🧪 Perturbation:    {args.perturbation}")
    print("-" * 60)
    print(f"✅ Mean Complementarity:  {mean_score:.4f}")
    print(f"📉 Std. Deviation:        {std_score:.4f}")
    print(f"📈 Number of Graphs:      {len(scores)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
