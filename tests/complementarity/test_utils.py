import numpy as np
from rings.complementarity.utils import maybe_normalize_diameter


class TestDiameterNorm:
    def test_nonzero_diameter(self):
        """Test that a matrix with a non-zero diameter gets normalized."""
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])

        expected = D / 3.0  # 3.0 is the max value
        result = maybe_normalize_diameter(D.copy())

        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_diameter(self):
        """Test that a matrix with zero diameter is returned unchanged."""
        D = np.zeros((3, 3))

        result = maybe_normalize_diameter(D.copy())

        np.testing.assert_array_equal(result, D)

    def test_single_element(self):
        """Test that a 1x1 matrix is handled correctly."""
        D = np.array([[0.0]])

        result = maybe_normalize_diameter(D.copy())

        np.testing.assert_array_equal(result, D)

    def test_negative_values(self):
        """Test handling of matrices with negative values."""
        D = np.array([[0.0, -1.0, 2.0], [-1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])

        expected = D / 3.0  # 3.0 is the max value
        result = maybe_normalize_diameter(D.copy())

        np.testing.assert_array_almost_equal(result, expected)

    def test_inplace_modification(self):
        """Test that the function modifies the array in-place."""
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])

        original = D.copy()
        result = maybe_normalize_diameter(D)

        # Check that the result is the same object as the input
        assert result is D

        # Check that the values have been modified
        assert not np.array_equal(D, original)
        np.testing.assert_array_almost_equal(D, original / 3.0)
