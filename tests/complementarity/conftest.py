import pytest
import numpy as np
from unittest.mock import MagicMock
from rings.complementarity.comparator import L11MatrixNormComparator
from rings.complementarity.functor import ComplementarityFunctor


# Helper function to standardize edge weights for tests
def ensure_float_weights(weights_dict):
    """Convert edge weights dictionary to a dict with only float values.
    This helps avoid inconsistent types between individual test runs and class test runs.
    """
    result = {}
    for edge, weight in weights_dict.items():
        if isinstance(weight, list):
            # If weight is a list, take its norm as a float
            result[edge] = float(np.linalg.norm(weight))
        else:
            result[edge] = float(weight)
    return result


# Helper function to check weight values consistently
def check_weights_approx(
    weights_dict, expected_value, message=None, tolerance=1e-6
):
    """Helper function to check weights consistently, handling both float and list weights."""
    weights_dict = ensure_float_weights(weights_dict)
    for weight in weights_dict.values():
        assert abs(weight - expected_value) < tolerance, (
            message or f"Expected weight {expected_value} but got {weight}"
        )


# Note: Removed the problematic autouse fixture that was causing conflicts
# with individual test patches. Tests now handle their own mocking.


@pytest.fixture(scope="function")
def mock_feature_metric():
    """Create a mock feature metric function."""
    return "euclidean"


@pytest.fixture(scope="function")
def mock_graph_metric():
    """Create a mock graph metric function."""
    return "shortest_path_distance"


@pytest.fixture(scope="function")
def mock_comparator():
    """Create a mock comparator class that returns 0.5 for all comparisons."""

    # Create a class with the required methods to mock the comparator
    class MockComparatorClass:
        def __init__(self, n_jobs=None, **kwargs):
            self.n_jobs = n_jobs
            self.kwargs = kwargs

        def __call__(self, D_X, D_G):
            return {"score": 0.5}

        @property
        def invalid_data(self):
            return {"score": np.nan}

    # Return the mock class
    return MagicMock(side_effect=MockComparatorClass)


@pytest.fixture(scope="function")
def functor(mock_feature_metric, mock_graph_metric, mock_comparator):
    """Create a ComplementarityFunctor instance for testing."""
    return ComplementarityFunctor(
        feature_metric=mock_feature_metric,
        graph_metric=mock_graph_metric,
        comparator=mock_comparator,
        n_jobs=1,
    )


@pytest.fixture(scope="function", autouse=False)
def reset_test_state():
    """Reset any shared state before and after each test."""
    # Setup before the test runs
    yield
    # Teardown after the test runs
    # No need to clear cached data as the functor is created fresh for each test
    # This fixture is kept for future use if needed
