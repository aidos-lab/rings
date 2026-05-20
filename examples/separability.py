"""
Separability Analysis using RINGS Framework

This script demonstrates how to:
1. Use the basic comparators (KS Test and Wilcoxon Test)
2. Apply the SeparabilityFunctor to compare multiple distributions
3. Visualize and interpret separability results

Usage (from root directory):
    python -m examples.separability --comparator ks --alpha 0.01

For more options:
    python -m examples.separability --help


Have your own performance distributions and want to execute a separability analysis?
Replace the distribution in `create_sample_distributions` function with your own and run the script.
"""

import numpy as np
import argparse
import time
from rings.separability.comparator import KSComparator, WilcoxonComparator
from rings.separability.functor import SeparabilityFunctor


def get_available_comparators():
    """
    Return a list of available comparators with descriptions.

    Returns:
        dict: Mapping from comparator name to description
    """
    return {
        "ks": "Kolmogorov-Smirnov test for comparing distributions",
        "wilcoxon": "Wilcoxon rank-sum test for comparing distributions",
    }


def create_comparator(name):
    """
    Create a comparator based on the given name.

    Args:
        name (str): Name of the comparator

    Returns:
        Comparator: A comparator object that can compare distributions
    """
    if name == "ks":
        return KSComparator()
    elif name == "wilcoxon":
        return WilcoxonComparator()
    else:
        raise ValueError(f"Unknown comparator: {name}")


def create_sample_distributions(seed=42):
    """
    Create sample distributions with different characteristics.

    Args:
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary of named distributions
    """
    np.random.seed(seed)
    return {
        "Original": np.random.normal(0, 1, 50),  # μ = 0
        "EmptyGraph": np.random.normal(0.5, 1, 50),  # μ = 0.5
        "CompleteGraph": np.random.normal(1.0, 1, 50),  # μ = 1.0
        "RandomGraph": np.random.normal(0, 1.5, 50),  # μ = 0, σ = 1.5
    }


def basic_comparator_example(seed=42):
    """
    Run a basic example using comparators directly.

    Args:
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)

    print("\n" + "=" * 60)
    print("📊 Basic Comparator Usage Example".center(60))
    print("=" * 60)

    # Sample data
    s1 = np.random.normal(0, 1, 100)  # From distribution 1
    s2 = np.random.normal(0.5, 1, 100)  # From distribution 2

    print("📈 Comparing two normal distributions:")
    print("   - Distribution 1: μ = 0, σ = 1, n = 100")
    print("   - Distribution 2: μ = 0.5, σ = 1, n = 100")
    print("-" * 60)

    # Using KS Comparator
    ks = KSComparator()
    ks_result = ks(s1, s2, n_hypotheses=5)
    print("🧪 KS Test Results:")
    print(f"   - Statistic: {ks_result['score']:.4f}")
    print(f"   - P-value:   {ks_result['pvalue']:.4f}")
    print(f"   - Adjusted:  {ks_result['pvalue_adjusted']:.4f}")
    print(f"   - Significant: {ks_result['significant']}")

    # Using Wilcoxon Comparator
    wc = WilcoxonComparator()
    wc_result = wc(s1, s2, n_hypotheses=5)
    print("\n🧮 Wilcoxon Test Results:")
    print(f"   - Statistic: {wc_result['score']:.4f}")
    print(f"   - P-value:   {wc_result['pvalue']:.4f}")
    print(f"   - Adjusted:  {wc_result['pvalue_adjusted']:.4f}")
    print(f"   - Significant: {wc_result['significant']}")


def run_separability_analysis(comparator_name, alpha, n_permutations, n_jobs, seed):
    """
    Run the separability analysis using the specified comparator.

    Args:
        comparator_name (str): Name of the comparator to use
        alpha (float): Significance level for statistical tests
        n_permutations (int): Number of permutations for the test
        n_jobs (int): Number of parallel jobs
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (results_df, results_list) containing the analysis results
    """
    print("\n" + "=" * 60)
    print("🔍 Separability Analysis".center(60))
    print("=" * 60)

    # Create sample distributions
    distributions = create_sample_distributions(seed)

    # Print distribution information
    print("📊 Analyzing distributions:")
    for name, dist in distributions.items():
        print(
            f"   - {name}: n = {len(dist)}, μ = {np.mean(dist):.2f}, σ = {np.std(dist):.2f}"
        )

    # Create comparator
    comparator = create_comparator(comparator_name)

    # Create SeparabilityFunctor
    functor = SeparabilityFunctor(
        comparator=comparator,
        n_jobs=n_jobs,
        alpha=alpha,
    )

    # Run the analysis
    start_time = time.time()
    print(f"\n⏳ Running analysis with {comparator_name.upper()} comparator...")
    print(f"   - Alpha: {alpha}")
    print(f"   - Permutations: {n_permutations}")

    # Get results as DataFrame
    results_df = functor.forward(
        distributions=distributions,
        n_permutations=n_permutations,
        random_state=seed,
        as_dataframe=True,
    )

    # Get results as list for more detailed reporting
    results_list = functor.forward(
        distributions=distributions,
        n_permutations=n_permutations,
        random_state=seed,
        as_dataframe=False,
    )

    duration = time.time() - start_time
    print(f"✅ Analysis completed in {duration:.2f} seconds")

    return results_df, results_list


def print_results_summary(results_df, results_list, comparator_name):
    """
    Print a summary of the results.

    Args:
        results_df: Results as DataFrame
        results_list: Results as list of dictionaries
        comparator_name (str): Name of the comparator used
    """
    # Count significant differences
    significant_count = sum(result["significant"] for result in results_list)
    total_comparisons = len(results_list)

    print("\n" + "=" * 60)
    print("📊 Results Summary".center(60))
    print("=" * 60)
    print(f"🧪 Comparator:        {comparator_name.upper()}")
    print(f"📈 Total Comparisons: {total_comparisons}")
    print(
        f"⚠️  Significant Diffs: {significant_count} ({significant_count / total_comparisons * 100:.1f}%)"
    )
    print("-" * 60)

    # Print DataFrame representation
    print("\n📋 Results Table:")
    print(results_df)

    # Print detailed results
    print("\n📝 Detailed Results:")
    for result in results_list:
        significant = "✅" if result["significant"] else "❌"
        print(
            f"{result['mode1']} vs {result['mode2']}: "
            f"stat={result['score']:.3f}, "
            f"p-value={result['pvalue']:.4f}, "
            f"adj.p={result['pvalue_adjusted']:.4f}, "
            f"{significant}"
        )


def main():
    """Main function to run the separability analysis.
    User inputs are handled via command line arguments:
        --comparator: Name of the comparator to use (default: ks)
        --alpha: Significance level for statistical tests (default: 0.01)
        --seed: Random seed for reproducibility (default: 42)
        --permutations: Number of permutations for the test (default: 1000)
        --n-jobs: Number of parallel jobs (-1 for all cores, default: 1)
        --skip-basic: Include --skip-basic argument to skip basic example, otherwise included by default
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute separability metrics for distributions"
    )

    # Get available comparators for help text
    comparators = get_available_comparators()
    comparator_help = "Available comparators:\n" + "\n".join(
        f"  {name}: {desc}" for name, desc in comparators.items()
    )

    parser.add_argument(
        "--comparator",
        type=str,
        default="ks",
        choices=list(comparators.keys()),
        help=f"Comparator to use for the separability analysis. {comparator_help}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for statistical tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of permutations for the test",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (-1 for all cores)",
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="Skip the basic comparator example",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Run basic example unless skipped
    if not args.skip_basic:
        basic_comparator_example(args.seed)

    # Run separability analysis
    results_df, results_list = run_separability_analysis(
        comparator_name=args.comparator,
        alpha=args.alpha,
        n_permutations=args.permutations,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    # Print results summary
    print_results_summary(results_df, results_list, args.comparator)


if __name__ == "__main__":
    main()
