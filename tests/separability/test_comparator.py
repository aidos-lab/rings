import pytest
import numpy as np
from rings.separability.comparator import KSComparator, WilcoxonComparator


class TestKSComparator:
    def test_ks_comparator_detects_difference(self):
        """KSComparator correctly identifies a significant difference between two clearly distinct distributions."""
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(5, 1, 100)
        comparator = KSComparator()
        result = comparator(x, y, n_permutations=500, alpha=0.05, random_state=123)
        assert result["significant"] is True
        assert result["method"] == "KS"
        assert 0 <= result["pvalue"] <= 1
        assert 0 <= result["pvalue_adjusted"] <= 1

    def test_ks_comparator_bonferroni_correction(self):
        """KSComparator applies Bonferroni correction to the p-value when multiple hypotheses are specified."""
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)
        comparator = KSComparator()
        n_hypotheses = 5
        result = comparator(
            x, y, n_permutations=200, n_hypotheses=n_hypotheses, random_state=42
        )
        assert result["pvalue_adjusted"] == min(1.0, result["pvalue"] * n_hypotheses)
        assert result["method"] == "KS"

    def test_ks_comparator_empty_input(self):
        """KSComparator handles empty input arrays gracefully."""
        comparator = KSComparator()
        x = []
        y = []
        with pytest.raises(ValueError):
            comparator(x, y, n_permutations=10, random_state=1)


class TestWilcoxonComparator:
    def test_wilcoxon_comparator_identical_samples(self):
        """WilcoxonComparator returns expected results when comparing two identical paired samples."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        comparator = WilcoxonComparator()
        result = comparator(x, y, n_permutations=100, alpha=0.05, random_state=123)
        assert result["significant"] is False
        assert result["method"] == "Wilcoxon"
        assert 0 <= result["pvalue"] <= 1
        assert 0 <= result["pvalue_adjusted"] <= 1

    def test_wilcoxon_comparator_unequal_length(self):
        """WilcoxonComparator raises a ValueError when input samples have different lengths."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        comparator = WilcoxonComparator()
        with pytest.raises(
            ValueError,
            match="Wilcoxon signed-rank test requires samples of equal length",
        ):
            comparator(x, y, n_permutations=10, random_state=1)

    def test_wilcoxon_comparator_random_state_consistency(self):
        """WilcoxonComparator produces consistent results when a fixed random_state is provided."""
        x = np.random.normal(0, 1, 20)
        y = np.random.normal(0, 1, 20)
        comparator = WilcoxonComparator()
        result1 = comparator(x, y, n_permutations=50, random_state=99)
        result2 = comparator(x, y, n_permutations=50, random_state=99)
        assert result1 == result2


# Source: rings/separability/comparator.py
