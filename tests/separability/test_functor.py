import pytest
import numpy as np
import pandas as pd

from rings.separability.functor import SeparabilityFunctor


class DummyComparator:
    def __init__(self):
        self.called_with = []

    def __call__(self, s1, s2, alpha=None, n_hypotheses=None, **kwargs):
        # Save call for inspection
        self.called_with.append(
            {
                "s1": s1,
                "s2": s2,
                "alpha": alpha,
                "n_hypotheses": n_hypotheses,
                **kwargs,
            }
        )
        # Return a dummy result
        return {
            "p_value": 0.005,
            "statistic": 1.23,
            "significant": alpha is not None and 0.005 < alpha,
        }


class TestSeparabilityFunctor:
    def test_forward_multiple_distributions_returns_dataframe(self):
        comparator = DummyComparator()
        functor = SeparabilityFunctor(comparator=comparator)
        distributions = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([4.0, 5.0, 6.0]),
            "C": np.array([7.0, 8.0, 9.0]),
        }
        result = functor.forward(distributions)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {
            "mode1",
            "mode2",
            "p_value",
            "statistic",
            "significant",
        }
        assert len(result) == 3  # 3 pairs for 3 distributions

    def test_bonferroni_correction_applied(self):
        class BonferroniComparator:
            def __init__(self):
                self.n_hypotheses_seen = []

            def __call__(self, s1, s2, alpha=None, n_hypotheses=None, **kwargs):
                self.n_hypotheses_seen.append(n_hypotheses)
                return {"p_value": 0.01, "statistic": 1.0, "significant": False}

        comparator = BonferroniComparator()
        functor = SeparabilityFunctor(comparator=comparator)
        distributions = {
            "A": np.array([1, 2, 3]),
            "B": np.array([4, 5, 6]),
            "C": np.array([7, 8, 9]),
            "D": np.array([10, 11, 12]),
        }
        functor.forward(distributions)
        # There should be 6 pairs for 4 distributions
        assert all(n == 6 for n in comparator.n_hypotheses_seen)

    def test_forward_returns_list_when_as_dataframe_false(self):
        comparator = DummyComparator()
        functor = SeparabilityFunctor(comparator=comparator)
        distributions = {
            "A": np.array([1, 2, 3]),
            "B": np.array([4, 5, 6]),
        }
        result = functor.forward(distributions, as_dataframe=False)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)
        assert len(result) == 1

    def test_forward_single_distribution_returns_empty(self):
        comparator = DummyComparator()
        functor = SeparabilityFunctor(comparator=comparator)
        distributions = {"A": np.array([1, 2, 3])}
        result_df = functor.forward(distributions)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
        result_list = functor.forward(distributions, as_dataframe=False)
        assert isinstance(result_list, list)
        assert result_list == []

    def test_forward_empty_input_returns_empty(self):
        comparator = DummyComparator()
        functor = SeparabilityFunctor(comparator=comparator)
        distributions = {}
        result_df = functor.forward(distributions)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
        result_list = functor.forward(distributions, as_dataframe=False)
        assert isinstance(result_list, list)
        assert result_list == []

    def test_forward_passes_kwargs_to_comparator(self):
        class KwargComparator:
            def __init__(self):
                self.kwargs_seen = []

            def __call__(self, s1, s2, alpha=None, n_hypotheses=None, **kwargs):
                self.kwargs_seen.append(kwargs)
                return {"p_value": 0.01, "statistic": 1.0, "significant": False}

        comparator = KwargComparator()
        functor = SeparabilityFunctor(comparator=comparator, custom_arg=123)
        distributions = {
            "A": np.array([1, 2, 3]),
            "B": np.array([4, 5, 6]),
        }
        functor.forward(distributions)
        # The custom_arg should be present in the kwargs passed to the comparator
        assert any("custom_arg" in kw for kw in comparator.kwargs_seen)

    def test_forward_raises_on_invalid_distribution_data(self):
        comparator = DummyComparator()
        functor = SeparabilityFunctor(comparator=comparator)
        # Non-numeric data in one distribution
        distributions = {
            "A": np.array([1, 2, 3]),
            "B": ["a", "b", "c"],
        }
        with pytest.raises(Exception):
            functor.forward(distributions)
