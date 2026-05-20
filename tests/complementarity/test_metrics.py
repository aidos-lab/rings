import pytest
import numpy as np
import networkx as nx
import sklearn.metrics
from unittest.mock import patch

from rings.complementarity.metrics import (
    lift_attributes,
    lift_graph,
    standard_feature_metrics,
    diffusion_distance,
    heat_kernel_distance,
    resistance_distance,
    shortest_path_distance,
    _compute_laplacian,
    _compute_psi,
    _compute_spectrum,
)


class TestLifts:
    def test_lift_attributes_with_standard_metric(self):
        """Test lifting attributes with standard sklearn metric."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = lift_attributes(X, metric="euclidean", n_jobs=1)

        # Calculate expected result manually
        expected = sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean")

        np.testing.assert_array_almost_equal(result, expected)

    def test_lift_attributes_with_unsupported_metric(self):
        """Test lifting attributes with unsupported metric raises error."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        with pytest.raises(RuntimeError, match="Unsupported metric"):
            lift_attributes(X, metric="unsupported_metric", n_jobs=1)

    @patch("rings.complementarity.metrics.standard_feature_metrics")
    def test_lift_attributes_delegation(self, mock_standard_metrics):
        """Test delegation to standard_feature_metrics for sklearn metrics."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_standard_metrics.return_value = "mocked_result"

        result = lift_attributes(X, metric="euclidean", n_jobs=1)

        mock_standard_metrics.assert_called_once()
        assert result == "mocked_result"

    def test_lift_graph_with_valid_metric(self):
        """Test lifting graph with valid metric."""
        G = nx.path_graph(3)  # Simple path graph with 3 nodes

        with patch(
            "rings.complementarity.metrics.shortest_path_distance"
        ) as mock_metric:
            mock_metric.return_value = "mocked_graph_metric"
            result = lift_graph(G, metric="shortest_path_distance")

            mock_metric.assert_called_once_with(G)
            assert result == "mocked_graph_metric"

    def test_lift_graph_with_empty_graph(self):
        """Test lifting an empty graph returns zero matrix."""
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])  # Add nodes but no edges

        result = lift_graph(G, metric="shortest_path_distance")

        expected = np.zeros((3, 3))
        np.testing.assert_array_equal(result, expected)

    def test_lift_graph_with_unsupported_metric(self):
        """Test lifting graph with unsupported metric raises error."""
        G = nx.path_graph(3)

        with pytest.raises(RuntimeError, match="Unsupported metric"):
            lift_graph(G, metric="unsupported_metric")


class TestMetrics:
    def test_standard_feature_metrics(self):
        """Test standard feature metrics calculation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        result = standard_feature_metrics(X, metric="euclidean")
        expected = sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean")

        np.testing.assert_array_almost_equal(result, expected)

    def test_diffusion_distance(self):
        """Test diffusion distance calculation."""
        # Create a simple graph
        G = nx.path_graph(4)

        # Test with default parameters
        result = diffusion_distance(G, num_steps=1)

        # Verify shape
        assert result.shape == (4, 4)
        # Verify symmetry
        assert np.allclose(result, result.T)
        # Verify diagonal is zero
        assert np.allclose(np.diag(result), 0)

    def test_diffusion_distance_with_isolated_nodes(self):
        """Test diffusion distance with isolated nodes returns NaN."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1)  # Node 2 is isolated

        result = diffusion_distance(G)

        assert np.isnan(result)

    def test_heat_kernel_distance(self):
        """Test heat kernel distance calculation."""
        G = nx.path_graph(4)

        result = heat_kernel_distance(G, num_steps=1)

        # Verify shape
        assert result.shape == (4, 4)
        # Verify symmetry
        assert np.allclose(result, result.T)
        # Verify diagonal is zero
        assert np.allclose(np.diag(result), 0)

    def test_resistance_distance(self):
        """Test resistance distance calculation."""
        G = nx.path_graph(4)

        result = resistance_distance(G)

        # Check shape
        assert result.shape == (4, 4)
        # Verify symmetry
        assert np.allclose(result, result.T)
        # Verify diagonal is zero
        assert np.allclose(np.diag(result), 0)

        # For a path graph, resistance distance should be equal to shortest path distance
        # for unweighted graphs
        expected = shortest_path_distance(G)
        np.testing.assert_array_almost_equal(result, expected)

    def test_shortest_path_distance(self):
        """Test shortest path distance calculation."""
        G = nx.path_graph(4)

        result = shortest_path_distance(G)

        # Expected distances for a path graph with 4 nodes (0-1-2-3)
        expected = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_shortest_path_distance_weighted(self):
        """Test shortest path distance with weighted edges."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)])

        result = shortest_path_distance(G, weight="weight")

        expected = np.array([[0, 1, 3, 6], [1, 0, 2, 5], [3, 2, 0, 3], [6, 5, 3, 0]])

        np.testing.assert_array_equal(result, expected)


class TestHelperFunctions:
    def test_compute_laplacian_unnormalized(self):
        """Test computation of unnormalized Laplacian."""
        # Simple adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        # Degree matrix
        D = np.diag(np.sum(A, axis=1))

        result = _compute_laplacian(A, D, norm=False)

        expected = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])

        np.testing.assert_array_equal(result, expected)

    def test_compute_laplacian_normalized_symmetric(self):
        """Test computation of normalized symmetric Laplacian."""
        # Simple adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        # Degree matrix
        D = np.diag(np.sum(A, axis=1))

        result = _compute_laplacian(A, D, norm=True, symmetric=True)

        # For this complete graph, the normalized symmetric Laplacian should be
        # I - D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        expected = np.eye(3) - D_inv_sqrt @ A @ D_inv_sqrt

        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_laplacian_normalized_asymmetric(self):
        """Test computation of normalized asymmetric Laplacian (random walk)."""
        # Simple adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        # Degree matrix
        D = np.diag(np.sum(A, axis=1))

        result = _compute_laplacian(A, D, norm=True, symmetric=False)

        # For random walk Laplacian: I - D^(-1) * A
        D_inv = np.diag(1.0 / np.diag(D))
        expected = np.eye(3) - D_inv @ A

        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_spectrum_symmetric(self):
        """Test computation of spectrum for symmetric matrices."""
        # Create a symmetric matrix
        L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])

        eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=True)

        # Check that eigenvalues are real
        assert np.allclose(np.imag(eigenvalues), 0)

        # Verify that L * v = lambda * v for each eigenvector v and corresponding eigenvalue lambda
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lambda_i = eigenvalues[i]
            np.testing.assert_array_almost_equal(L @ v, lambda_i * v)

    def test_compute_psi(self):
        """Test computation of psi matrix."""
        # Simple Laplacian matrix
        L = np.array([[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]])

        t = 2
        psi = _compute_psi(L, t, symmetric=True)

        # Verify shape
        assert psi.shape == L.shape

        # Manually compute expected result
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        eigenvalues_power = np.power(eigenvalues, t)
        expected = eigenvalues_power * eigenvectors

        np.testing.assert_array_almost_equal(psi, expected)
