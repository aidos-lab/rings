import numpy as np
from scipy.stats import ks_2samp, wilcoxon
from typing import Dict, List, Union, Optional
from tqdm import tqdm


class KSComparator:
    """
    Comparator that uses the Kolmogorov-Smirnov test to compare two distributions.

    This class implements a statistical test based on the Kolmogorov-Smirnov (KS)
    statistic to determine if two samples come from the same distribution. It uses
    permutation testing to empirically determine significance, making it suitable
    for small sample sizes where asymptotic approximations may not be reliable.

    Methods
    -------
    __call__(x, y, n_permutations=10_000, alpha=0.01, n_hypotheses=1, random_state=42, **kwargs)
        Compare two samples using Kolmogorov-Smirnov test with permutation testing.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.separability.comparator import KSComparator
    >>>
    >>> # Create two samples from different distributions
    >>> x = np.random.normal(0, 1, 20)  # Standard normal
    >>> y = np.random.normal(1, 1, 20)  # Shifted normal
    >>>
    >>> # Compare distributions with KS test
    >>> comparator = KSComparator()
    >>> result = comparator(x, y, n_permutations=1000, alpha=0.05)
    >>>
    >>> # Interpret results
    >>> print(f"KS statistic: {result['score']}")
    >>> print(f"p-value: {result['pvalue']}")
    >>> print(f"Significant: {result['significant']}")

    Notes
    -----
    The Kolmogorov-Smirnov test measures the maximum difference between the
    empirical cumulative distribution functions of two samples. It is sensitive
    to differences in both location and shape of distributions.
    """

    def __call__(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        n_permutations: int = 10_000,
        alpha: float = 0.01,
        n_hypotheses: int = 1,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> Dict:
        """
        Compare two samples using Kolmogorov-Smirnov test with permutation testing.

        Parameters
        ----------
        x : Union[List[float], np.ndarray]
            First sample of measurements
        y : Union[List[float], np.ndarray]
            Second sample of measurements
        n_permutations : int, optional
            Number of permutations for the test (default: 10,000)
        alpha : float, optional
            Significance level (default: 0.01)
        n_hypotheses : int, optional
            Number of hypotheses being tested (for Bonferroni correction)
        random_state : Optional[int], optional
            Seed for reproducibility
        **kwargs : dict, optional
            Additional keyword arguments (not used)

        Returns
        -------
        dict
            Dictionary with keys:
            - score: The KS statistic observed in the original data
            - pvalue: The p-value from the permutation test
            - pvalue_adjusted: The Bonferroni-corrected p-value
            - significant: Whether the adjusted p-value is less than alpha
            - method: "KS" (identifier for this test)

        Raises
        ------
        ValueError
            If either input array is empty
        """
        # Convert inputs to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Check for empty arrays
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays must not be empty for KS test")

        # Check if arrays are identical
        if np.array_equal(x, y):
            return {
                "score": 0.0,
                "pvalue": 1.0,
                "pvalue_adjusted": 1.0,
                "significant": False,
                "method": "KS",
            }

        # Initialize RNG
        rng = np.random.RandomState(random_state)

        # Calculate observed statistic
        observed_result = ks_2samp(x, y)
        observed_statistic = observed_result.statistic

        # Prepare for permutation test
        combined = np.concatenate((x, y))
        n1 = len(x)

        # Run permutation test
        permuted_statistics = np.zeros(n_permutations)

        for i in tqdm(range(n_permutations), desc="KS Permutation Test"):
            # Shuffle the combined data
            shuffled = rng.permutation(combined)

            # Split into two groups of original sizes
            perm_s1 = shuffled[:n1]
            perm_s2 = shuffled[n1:]

            # Compute statistic on permuted data
            perm_result = ks_2samp(perm_s1, perm_s2)
            permuted_statistics[i] = perm_result.statistic

        # Calculate p-value (two-sided)
        pvalue = np.mean(permuted_statistics >= observed_statistic)

        # Apply Bonferroni correction
        pvalue_adjusted = min(1.0, pvalue * n_hypotheses)

        # Convert numpy boolean to Python boolean for consistent behavior
        is_significant = bool(pvalue_adjusted < alpha)

        return {
            "score": observed_statistic,
            "pvalue": pvalue,
            "pvalue_adjusted": pvalue_adjusted,
            "significant": is_significant,
            "method": "KS",
        }


class WilcoxonComparator:
    """
    Comparator that uses the Wilcoxon signed-rank test to compare two distributions.

    This class implements a statistical test based on the Wilcoxon signed-rank
    test for paired samples. It uses permutation testing to empirically determine
    significance, making it suitable for small sample sizes or when assumptions
    for parametric tests are not met.

    Methods
    -------
    __call__(x, y, n_permutations=10_000, alpha=0.01, n_hypotheses=1, random_state=42, **kwargs)
        Compare two samples using Wilcoxon signed-rank test with permutation testing.

    Examples
    --------
    >>> import numpy as np
    >>> from rings.separability.comparator import WilcoxonComparator
    >>>
    >>> # Create paired samples (must be equal length)
    >>> np.random.seed(42)
    >>> x = np.random.normal(0, 1, 20)
    >>> y = x + np.random.normal(0.5, 0.5, 20)  # Same as x with shift + noise
    >>>
    >>> # Compare distributions
    >>> comparator = WilcoxonComparator()
    >>> result = comparator(x, y, n_permutations=1000, alpha=0.05)
    >>>
    >>> # Interpret results
    >>> print(f"Wilcoxon statistic: {result['score']}")
    >>> print(f"p-value: {result['pvalue']}")
    >>> print(f"Significant: {result['significant']}")

    Notes
    -----
    The Wilcoxon signed-rank test compares paired samples by ranking the
    absolute differences between pairs and summing the ranks of positive
    differences. Unlike the t-test, it does not require the differences
    to be normally distributed.

    This implementation requires paired samples of equal length.
    """

    def __call__(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        n_permutations: int = 10_000,
        alpha: float = 0.01,
        n_hypotheses: int = 1,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> Dict:
        """
        Compare two samples using Wilcoxon signed-rank test with permutation testing.

        Parameters
        ----------
        x : Union[List[float], np.ndarray]
            First sample of measurements
        y : Union[List[float], np.ndarray]
            Second sample of measurements
        n_permutations : int, optional
            Number of permutations for the test (default: 10,000)
        alpha : float, optional
            Significance level (default: 0.01)
        n_hypotheses : int, optional
            Number of hypotheses being tested (for Bonferroni correction)
        random_state : Optional[int], optional
            Seed for reproducibility
        **kwargs : dict, optional
            Additional keyword arguments (not used)

        Returns
        -------
        dict
            Dictionary with keys:
            - score: The Wilcoxon statistic observed in the original data
            - pvalue: The p-value from the permutation test
            - pvalue_adjusted: The Bonferroni-corrected p-value
            - significant: Whether the adjusted p-value is less than alpha
            - method: "Wilcoxon" (identifier for this test)
        """
        # Convert inputs to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Initialize RNG
        rng = np.random.RandomState(random_state)

        # Verify arrays have the same length for Wilcoxon paired test
        if len(x) != len(y):
            raise ValueError(
                "Wilcoxon signed-rank test requires samples of equal length"
            )

        # Check if arrays are identical or have all zero differences
        # which would cause division by zero in the wilcoxon test
        if np.array_equal(x, y) or np.all(x - y == 0):
            return {
                "score": 0.0,
                "pvalue": 1.0,
                "pvalue_adjusted": 1.0,
                "significant": False,
                "method": "Wilcoxon",
            }

        # Calculate observed statistic
        observed_result = wilcoxon(x, y, alternative="two-sided")
        observed_statistic = observed_result.statistic

        # Prepare for permutation test
        combined = np.vstack((x, y)).T  # Paired data
        n_pairs = len(combined)

        # Run permutation test
        permuted_statistics = np.zeros(n_permutations)

        for i in tqdm(range(n_permutations), desc="Wilcoxon Permutation Test"):
            # For each pair, randomly swap or don't swap the elements
            perm_combined = combined.copy()
            swap_indices = rng.choice([True, False], size=n_pairs)

            if swap_indices.any():
                (
                    perm_combined[swap_indices, 0],
                    perm_combined[swap_indices, 1],
                ) = (
                    perm_combined[swap_indices, 1],
                    perm_combined[swap_indices, 0],
                )

            # Split into two groups
            perm_s1 = perm_combined[:, 0]
            perm_s2 = perm_combined[:, 1]

            # Compute statistic on permuted data
            perm_result = wilcoxon(perm_s1, perm_s2, alternative="two-sided")
            permuted_statistics[i] = perm_result.statistic

        # Calculate p-value (two-sided)
        # For Wilcoxon, the direction matters, so we compare absolute deviations
        # from the mean of the permutation distribution
        mean_perm_stat = np.mean(permuted_statistics)
        observed_dev = np.abs(observed_statistic - mean_perm_stat)
        perm_devs = np.abs(permuted_statistics - mean_perm_stat)

        pvalue = np.mean(perm_devs >= observed_dev)

        # Apply Bonferroni correction
        pvalue_adjusted = min(1.0, pvalue * n_hypotheses)

        # Convert numpy boolean to Python boolean for consistent behavior
        is_significant = bool(pvalue_adjusted < alpha)

        return {
            "score": observed_statistic,
            "pvalue": pvalue,
            "pvalue_adjusted": pvalue_adjusted,
            "significant": is_significant,
            "method": "Wilcoxon",
        }
