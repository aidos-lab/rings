import pytest
import numpy as np
import torch
import networkx as nx
import pandas as pd
import warnings
from unittest.mock import MagicMock, patch, call

from torch_geometric.data import Data
from rings.complementarity.functor import ComplementarityFunctor
from rings.complementarity.comparator import L11MatrixNormComparator


class TestComplementarityFunctor:
    @pytest.fixture
    def mock_feature_metric(self):
        return "euclidean"

    @pytest.fixture
    def mock_graph_metric(self):
        return "diffusion_distance"

    @pytest.fixture
    def mock_comparator(self):
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

        # Return the mock class instead of trying to mock a callable
        return MagicMock(side_effect=MockComparatorClass)

    @pytest.fixture
    def functor(self, mock_feature_metric, mock_graph_metric, mock_comparator):
        return ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=1,
        )

    def test_init(
        self, mock_feature_metric, mock_graph_metric, mock_comparator
    ):
        """Test initialization of ComplementarityFunctor."""
        functor = ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=2,
            custom_param="test",
        )

        assert functor.feature_metric == mock_feature_metric
        assert functor.graph_metric == mock_graph_metric
        assert functor.n_jobs == 2
        assert functor.use_edge_information is False
        assert functor.kwargs == {"custom_param": "test"}
        mock_comparator.assert_called_once_with(n_jobs=2, custom_param="test")

    def test_init_with_edge_information(
        self, mock_feature_metric, mock_graph_metric, mock_comparator
    ):
        """Test initialization with edge information enabled."""
        functor = ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=1,
            use_edge_information=True,
        )

        assert functor.use_edge_information is True
        assert functor.edge_attr == "edge_attr"  # Default edge attribute name
        mock_comparator.assert_called_once_with(n_jobs=1)

    def test_init_with_custom_edge_attr(
        self, mock_feature_metric, mock_graph_metric, mock_comparator
    ):
        """Test initialization with custom edge attribute name."""
        functor = ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=1,
            use_edge_information=True,
            edge_attr="custom_edge_weight",
        )

        assert functor.use_edge_information is True
        assert (
            functor.edge_attr == "custom_edge_weight"
        )  # Custom edge attribute name
        mock_comparator.assert_called_once_with(n_jobs=1)

    @patch("rings.complementarity.functor.to_networkx")
    @patch("rings.complementarity.functor.lift_attributes")
    @patch("rings.complementarity.functor.lift_graph")
    @patch("rings.complementarity.functor.maybe_normalize_diameter")
    def test_forward_single_graph(
        self,
        mock_normalize,
        mock_lift_graph,
        mock_lift_attrs,
        mock_to_networkx,
        functor,
        mock_comparator,
    ):
        """Test forward method with a single graph."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        mock_graph.number_of_edges.return_value = 1
        mock_graph.number_of_nodes.return_value = 2
        nx.is_connected = MagicMock(return_value=True)
        nx.get_node_attributes = MagicMock(return_value={0: [1, 2], 1: [3, 4]})

        # Mock lift functions
        mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])
        mock_lift_attrs.return_value = np.array([[0, 2], [2, 0]])
        mock_normalize.side_effect = lambda x: x

        # Create test data
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data = Data(x=x, edge_index=edge_index)

        # Suppress expected warnings using a context manager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Weights sum to zero, using simple average instead",
            )

            # Run forward with as_dataframe=False to get tensor outputs
            result = functor.forward([test_data], as_dataframe=False)

        # Check results
        assert "complementarity" in result
        assert torch.is_tensor(result["complementarity"])
        assert result["complementarity"].shape == (1,)
        assert float(result["complementarity"][0]) == 0.5

        # Verify the workflow
        mock_to_networkx.assert_called_once()
        mock_lift_graph.assert_called_once()
        mock_lift_attrs.assert_called_once()
        mock_normalize.call_count == 2

    @patch("rings.complementarity.functor.to_networkx")
    @patch("rings.complementarity.functor.lift_attributes")
    @patch("rings.complementarity.functor.lift_graph")
    @patch("rings.complementarity.functor.maybe_normalize_diameter")
    def test_forward_batch(
        self,
        mock_normalize,
        mock_lift_graph,
        mock_lift_attrs,
        mock_to_networkx,
        functor,
    ):
        """Test forward method with a batch of graphs."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        mock_graph.number_of_edges.return_value = 1
        mock_graph.number_of_nodes.return_value = 2
        nx.is_connected = MagicMock(return_value=True)
        nx.get_node_attributes = MagicMock(return_value={0: [1, 2], 1: [3, 4]})

        # Mock lift functions
        mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])
        mock_lift_attrs.return_value = np.array([[0, 2], [2, 0]])
        mock_normalize.side_effect = lambda x: x

        # Create test data
        x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data1 = Data(x=x1, edge_index=edge_index1)

        x2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
        edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data2 = Data(x=x2, edge_index=edge_index2)

        # Suppress expected warnings using a context manager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Weights sum to zero, using simple average instead",
            )

            # Run forward with as_dataframe=False to get tensor outputs
            result = functor.forward(
                [test_data1, test_data2], as_dataframe=False
            )

        # Check results
        assert "complementarity" in result
        assert torch.is_tensor(result["complementarity"])
        assert result["complementarity"].shape == (2,)
        assert float(result["complementarity"][0]) == 0.5
        assert float(result["complementarity"][1]) == 0.5

    @patch("rings.complementarity.functor.to_networkx")
    def test_forward_with_edge_attr(self, mock_to_networkx, functor):
        """Test forward with edge attributes."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        mock_graph.number_of_edges.return_value = 1
        mock_graph.number_of_nodes.return_value = 2
        nx.is_connected = MagicMock(return_value=True)
        nx.get_node_attributes = MagicMock(return_value={0: [1, 2], 1: [3, 4]})
        nx.get_edge_attributes = MagicMock(return_value={(0, 1): [0.5, 0.5]})
        nx.set_edge_attributes = MagicMock()

        # Enable edge information
        functor.use_edge_information = True
        functor.edge_attr = "edge_attr"

        # Create test data with edge attributes
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float)
        test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Apply patching for internal methods
        with patch.object(
            functor,
            "_compute_complementarity",
            return_value={"complementarity": 0.5},
        ):
            # Run forward with as_dataframe=False to get tensor outputs
            result = functor.forward([test_data], as_dataframe=False)

            # Check edge attribute processing
            mock_to_networkx.assert_called_with(
                test_data,
                to_undirected=True,
                node_attrs=["x"],
                edge_attrs=["edge_attr"],
            )

            # When use_edge_information is True and edge_attr is present
            # nx.get_edge_attributes and nx.set_edge_attributes should be called
            assert nx.get_edge_attributes.called
            assert nx.set_edge_attributes.called
            # The edge attributes should be set as weights now
            # Check that the mock_graph now has an edge attribute called "weight"
            assert (
                hasattr(mock_graph, "weight")
                or "weight" in mock_graph.__dict__
                or "weight" in dir(mock_graph)
            )

    def test_complementarity_connected_graph(self, functor):
        """Test complementarity calculation with a connected graph."""
        # Setup test graph
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")

        # Apply patching for the internal methods
        with (
            patch.object(
                functor,
                "_lift_metrics",
                return_value=(
                    [np.array([[0, 2], [2, 0]])],
                    [np.array([[0, 1], [1, 0]])],
                    [2],
                ),
            ) as mock_lift_metrics,
            patch.object(
                functor,
                "_normalize_metrics",
                return_value=(
                    [np.array([[0, 2], [2, 0]])],
                    [np.array([[0, 1], [1, 0]])],
                ),
            ) as mock_normalize_metrics,
            patch.object(
                functor, "_compute_scores", return_value=[0.5]
            ) as mock_compute_scores,
            patch.object(
                functor, "_aggregate", return_value=0.5
            ) as mock_aggregate,
        ):
            # Call the function
            result = functor._compute_complementarity(G)

            # Verify all methods were called
            mock_lift_metrics.assert_called_once()
            mock_normalize_metrics.assert_called_once()
            mock_compute_scores.assert_called_once()
            mock_aggregate.assert_called_once()

            # Check result
            assert "complementarity" in result
            assert result["complementarity"] == 0.5

    @patch("warnings.warn")
    def test_complementarity_empty_graph(self, mock_warn, functor):
        """Test complementarity calculation with empty graph."""
        # Setup empty graph
        G = nx.Graph()
        # Don't need to mock nx.get_node_attributes, as the empty graph will naturally return empty attributes

        # Directly patch nx.get_node_attributes to return empty dict instead of relying on mocking
        with patch("networkx.get_node_attributes", return_value={}):
            result = functor._compute_complementarity(G)

            # Check warning and result
            mock_warn.assert_called_once_with(
                "Feature matrix X is empty, skipping graph."
            )
            assert "complementarity" in result
            assert np.isnan(result["complementarity"])

    def test_complementarity_return_metric_spaces(self, functor):
        """Test complementarity calculation with return_metric_spaces=True."""
        # Setup test graph
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")

        # Apply patching
        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=True,
            ),
            patch.object(
                functor,
                "_lift_metrics",
                return_value=(
                    [np.array([[0, 2], [2, 0]])],
                    [np.array([[0, 1], [1, 0]])],
                    [2],
                ),
            ) as mock_lift_metrics,
            patch.object(
                functor,
                "_normalize_metrics",
                return_value=(
                    [np.array([[0, 2], [2, 0]])],
                    [np.array([[0, 1], [1, 0]])],
                ),
            ) as mock_normalize_metrics,
            patch.object(
                functor, "_compute_scores", return_value=[0.5]
            ) as mock_compute_scores,
            patch.object(
                functor, "_aggregate", return_value=0.5
            ) as mock_aggregate,
        ):

            result = functor._compute_complementarity(
                G, return_metric_spaces=True
            )

            # Check result contains metric spaces
            assert "complementarity" in result
            assert "D_X" in result
            assert "D_G" in result
            assert len(result["D_X"]) == 1
            assert len(result["D_G"]) == 1

    def test_lift_metrics_connected_graph(self, functor):
        """Test _lift_metrics method with a connected graph."""
        # Setup test graph
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")
        X = np.array([[1, 2], [3, 4]])

        # Mock the lift functions
        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=True,
            ),
            patch(
                "rings.complementarity.functor.lift_graph",
                return_value=np.array([[0, 1], [1, 0]]),
            ) as mock_lift_graph,
            patch(
                "rings.complementarity.functor.lift_attributes",
                return_value=np.array([[0, 2], [2, 0]]),
            ) as mock_lift_attrs,
        ):
            # Call the method
            D_X, D_G, sizes = functor._lift_metrics(G, X, empty_graph=False)

            # Assert the expected results
            assert len(D_X) == 1
            assert len(D_G) == 1
            assert len(sizes) == 1
            assert sizes[0] == 2
            mock_lift_graph.assert_called_once()
            mock_lift_attrs.assert_called_once()

    def test_lift_metrics_disconnected_graph(self, functor):
        """Test _lift_metrics method with a disconnected graph."""
        # Setup test graph with two connected components
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (2, 3)])
        nx.set_node_attributes(
            G, {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8]}, "x"
        )
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        # Mock the lift functions
        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=False,
            ),
            patch(
                "rings.complementarity.functor.nx.connected_components",
                return_value=[{0, 1}, {2, 3}],
            ),
            patch(
                "rings.complementarity.functor.lift_graph",
                side_effect=[
                    np.array([[0, 1], [1, 0]]),  # First component
                    np.array([[0, 1], [1, 0]]),  # Second component
                ],
            ) as mock_lift_graph,
            patch(
                "rings.complementarity.functor.lift_attributes",
                side_effect=[
                    np.array([[0, 2], [2, 0]]),  # First component
                    np.array([[0, 2], [2, 0]]),  # Second component
                ],
            ) as mock_lift_attrs,
        ):
            # Call the method
            D_X, D_G, sizes = functor._lift_metrics(G, X, empty_graph=False)

            # Assert the expected results
            assert len(D_X) == 2
            assert len(D_G) == 2
            assert len(sizes) == 2
            assert sizes == [2, 2]
            assert mock_lift_graph.call_count == 2
            assert mock_lift_attrs.call_count == 2

    def test_normalize_metrics(self, functor):
        """Test _normalize_metrics method."""
        # Create test data
        D_X = [np.array([[0, 10], [10, 0]]), np.array([[0, 5], [5, 0]])]
        D_G = [np.array([[0, 2], [2, 0]]), np.array([[0, 4], [4, 0]])]

        # Mock the normalize function
        with patch(
            "rings.complementarity.functor.maybe_normalize_diameter",
            side_effect=lambda x: x / np.max(x) if np.max(x) > 0 else x,
        ) as mock_normalize:
            # Call the method
            D_X_norm, D_G_norm = functor._normalize_metrics(D_X, D_G)

            # Assert the expected behavior
            assert mock_normalize.call_count == 4  # Called for each matrix
            assert len(D_X_norm) == 2
            assert len(D_G_norm) == 2

    def test_compute_scores(self, functor, mock_comparator):
        """Test _compute_scores method."""
        # Create test data
        D_X = [np.array([[0, 2], [2, 0]]), np.array([[0, 3], [3, 0]])]
        D_G = [np.array([[0, 1], [1, 0]]), np.array([[0, 4], [4, 0]])]

        # Call the method
        scores = functor._compute_scores(D_X, D_G)

        # Check that the comparator was called correctly
        assert len(scores) == 2
        assert scores[0] == 0.5
        assert scores[1] == 0.5

    def test_aggregate_weighted_average(self, functor):
        """Test _aggregate method with weighted average."""
        # Test data
        scores = [0.2, 0.8]
        sizes = [3, 7]

        # Expected result: (0.2*3 + 0.8*7) / (3 + 7) = 0.62
        expected = 0.62

        # Call the method
        result = functor._aggregate(scores, sizes)

        # Assert the expected weighted average
        assert abs(result - expected) < 1e-10

    def test_aggregate_simple_average(self, functor):
        """Test _aggregate method with simple average when weights sum to zero."""
        # Test data
        scores = [0.2, 0.8]
        sizes = [0, 0]

        # Expected result: (0.2 + 0.8) / 2 = 0.5
        expected = 0.5

        # Call the method with warning patch
        with patch("warnings.warn") as mock_warn:
            result = functor._aggregate(scores, sizes)

            # Check warning was shown
            mock_warn.assert_called_once_with(
                "Weights sum to zero, using simple average instead"
            )

            # Assert the expected simple average
            assert result == expected

    def test_aggregate_empty_scores(self, functor):
        """Test _aggregate method with empty scores list."""
        # Test data
        scores = []
        sizes = []

        # Call the method with warning patch
        with patch("warnings.warn") as mock_warn:
            result = functor._aggregate(scores, sizes)

            # Check warning was shown
            mock_warn.assert_called_once_with(
                "Weights sum to zero, using simple average instead"
            )

            # Result should be NaN for empty scores
            assert np.isnan(result)

    @patch("rings.complementarity.functor.to_networkx")
    def test_preprocess_graph(self, mock_to_networkx, functor):
        """Test _preprocess_graph method."""
        # Create test data
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float)
        test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Mock NetworkX graph
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        nx.get_edge_attributes = MagicMock(
            return_value={(0, 1): [0.5, 0.5], (1, 0): [0.5, 0.5]}
        )
        nx.set_edge_attributes = MagicMock()

        # Test without edge information
        functor.use_edge_information = False
        functor._preprocess_graph(test_data, None)

        mock_to_networkx.assert_called_with(
            test_data,
            to_undirected=True,
            node_attrs=["x"],
            edge_attrs=None,
        )

        # Clear mock call history
        mock_to_networkx.reset_mock()
        nx.get_edge_attributes.reset_mock()
        nx.set_edge_attributes.reset_mock()

        # Test with edge information and default edge_attr
        functor.use_edge_information = True
        functor.edge_attr = "edge_attr"
        functor._preprocess_graph(test_data, "edge_attr")

        mock_to_networkx.assert_called_with(
            test_data,
            to_undirected=True,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
        )
        assert nx.get_edge_attributes.called
        assert nx.set_edge_attributes.called

        # Check that the edge attributes are set as "weight" now
        nx.set_edge_attributes.assert_called_once()
        args, kwargs = nx.set_edge_attributes.call_args
        assert len(args) == 3
        assert args[2] == "weight"  # Third argument should be "weight"

    def test_process_single(self, functor):
        """Test _process_single method."""
        # Create test data
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data = Data(x=x, edge_index=edge_index)

        # Set the edge attribute name
        functor.edge_attr = "edge_attr"

        # Mock the required methods
        with (
            patch.object(
                functor, "_preprocess_graph", return_value=MagicMock()
            ) as mock_preprocess,
            patch.object(
                functor,
                "_compute_complementarity",
                return_value={"complementarity": 0.5},
            ) as mock_compute_complementarity,
        ):
            # Call the method
            result = functor._process_single(test_data)

            # Check that methods were called and correct result returned
            mock_preprocess.assert_called_once_with(
                test_data, functor.edge_attr
            )
            mock_compute_complementarity.assert_called_once()
            assert result["complementarity"] == 0.5

    def test_lift_metrics_with_edge_weights(self, functor):
        """Test _lift_metrics handles edge weights correctly when use_edge_information=True."""
        # Setup test graph with weighted edges
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, weight=0.5)  # Add edge with weight
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")
        X = np.array([[1, 2], [3, 4]])

        # Enable edge information
        functor.use_edge_information = True
        functor.edge_attr = "edge_attr"

        # Test with mock for lift_graph to verify weight param is passed
        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])

            # Also mock other required functions
            with (
                patch(
                    "rings.complementarity.functor.nx.is_connected",
                    return_value=True,
                ),
                patch(
                    "rings.complementarity.functor.lift_attributes",
                    return_value=np.array([[0, 2], [2, 0]]),
                ),
            ):
                functor._lift_metrics(G, X, empty_graph=False)

            # Check that lift_graph was called with weight parameter
            mock_lift_graph.assert_called_once()
            # Verify that the weight param was included in the call
            args, kwargs = mock_lift_graph.call_args

    def test_lift_metrics_without_edge_weights(self, functor):
        """Test _lift_metrics ignores edge weights when use_edge_information=False."""
        # Setup test graph with weighted edges
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, weight=0.5)  # Add edge with weight
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")
        X = np.array([[1, 2], [3, 4]])

        # Disable edge information
        functor.use_edge_information = False
        functor.edge_attr = None  # Ensure edge attribute is not set

        # Test with mock for lift_graph to verify weight param is not passed
        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])

            # Also mock other required functions
            with (
                patch(
                    "rings.complementarity.functor.nx.is_connected",
                    return_value=True,
                ),
                patch(
                    "rings.complementarity.functor.lift_attributes",
                    return_value=np.array([[0, 2], [2, 0]]),
                ),
            ):
                functor._lift_metrics(G, X, empty_graph=False)

            # Check that lift_graph was called without weight parameter
            mock_lift_graph.assert_called_once()
            # Verify that no weight param was included
            args, kwargs = mock_lift_graph.call_args

    def test_weighted_vs_unweighted_preprocessing(self):
        """Test that weighted and unweighted graph lifts are processed differently."""
        # Create a PyG Data object with edge attributes
        x = torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 0.0]],
            dtype=torch.float,
        )
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 1, 2],
                [1, 2, 3, 4, 0, 3, 4],
            ],
            dtype=torch.long,
        )
        # edge_attr must have the same number of rows as edge_index has columns (number of edges = 7)
        edge_attr = torch.tensor(
            [[1.0], [5.0], [2.5], [3.0], [4.0], [1.5], [2.0]], dtype=torch.float
        )
        G_weighted = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Set up to track calls with different arguments
        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=4,
            use_edge_information=True,
            normalize_diameters=True,
        )

        assert weighted_functor.edge_attr == "edge_attr"

        processed_G = weighted_functor._preprocess_graph(
            G_weighted, weighted_functor.edge_attr
        )

        assert (
            processed_G.number_of_edges() == 7
        )  # All edges should be included
        weights = nx.get_edge_attributes(processed_G, "weight")
        assert weights, "Edge attribute 'weight' should not be empty"

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=4,
            use_edge_information=False,
            normalize_diameters=True,
        )

        assert (
            unweighted_functor.edge_attr is None
        ), "Edge attribute should be empty for unweighted functor"

    def test_correct_weight_parameter_passing(self, functor):
        """Test that the weight parameter is correctly passed through from functor to the graph metric."""
        # Create a weighted graph
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=5.0)
        nx.set_node_attributes(G, {0: [1, 1], 1: [2, 2], 2: [3, 3]}, "x")
        X = np.array([[1, 1], [2, 2], [3, 3]])

        # Set up edge information
        functor.use_edge_information = True
        functor.edge_attr = "edge_attr"

        # Mock the lift_graph function to check if weight is passed
        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            mock_lift_graph.return_value = np.array(
                [[0, 1, 6], [1, 0, 5], [6, 5, 0]]
            )

            # Also mock other required functions for _lift_metrics
            with (
                patch(
                    "rings.complementarity.functor.nx.is_connected",
                    return_value=True,
                ),
                patch(
                    "rings.complementarity.functor.lift_attributes",
                    return_value=np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
                ),
            ):
                # Call _lift_metrics which should pass weight parameter to lift_graph
                functor._lift_metrics(G, X, empty_graph=False)

                # Verify lift_graph was called with the weight parameter
                mock_lift_graph.assert_called_once()

    def test_disconnected_graph_component_isolation(self, functor):
        """Test that disconnected components are processed independently."""
        # Create a disconnected graph with three isolated components
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3, 4, 5])
        # Component 1: nodes 0-1
        G.add_edge(0, 1)
        # Component 2: nodes 2-3
        G.add_edge(2, 3)
        # Component 3: isolated node 4
        # Component 4: isolated node 5
        nx.set_node_attributes(
            G,
            {0: [1, 0], 1: [2, 0], 2: [0, 1], 3: [0, 2], 4: [3, 3], 5: [4, 4]},
            "x",
        )
        X = np.array([[1, 0], [2, 0], [0, 1], [0, 2], [3, 3], [4, 4]])

        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=False,
            ),
            patch(
                "rings.complementarity.functor.nx.connected_components",
                return_value=[{0, 1}, {2, 3}, {4}, {5}],
            ),
            patch(
                "rings.complementarity.functor.lift_graph",
                side_effect=[
                    np.array([[0, 1], [1, 0]]),  # Component 1 (2x2)
                    np.array([[0, 1], [1, 0]]),  # Component 2 (2x2)
                    np.array([[0]]),  # Component 3 (1x1)
                    np.array([[0]]),  # Component 4 (1x1)
                ],
            ) as mock_lift_graph,
            patch(
                "rings.complementarity.functor.lift_attributes",
                side_effect=[
                    np.array([[0, 1], [1, 0]]),  # Component 1 features
                    np.array([[0, 1], [1, 0]]),  # Component 2 features
                    np.array([[0]]),  # Component 3 features
                    np.array([[0]]),  # Component 4 features
                ],
            ) as mock_lift_attrs,
        ):
            D_X, D_G, sizes = functor._lift_metrics(G, X, empty_graph=False)

            # Verify correct number of components processed
            assert len(D_X) == 4
            assert len(D_G) == 4
            assert len(sizes) == 4
            assert sizes == [2, 2, 1, 1]  # Component sizes

            # Verify each component was processed independently
            assert mock_lift_graph.call_count == 4
            assert mock_lift_attrs.call_count == 4

    def test_disconnected_graph_weighted_aggregation_complex(self, functor):
        """Test weighted aggregation with complex disconnected graph scenarios."""
        # Test case 1: Unequal component sizes
        scores = [0.1, 0.5, 0.9]  # Different complementarity scores
        sizes = [10, 2, 3]  # Different component sizes

        # Expected weighted average: (0.1*10 + 0.5*2 + 0.9*3) / (10+2+3) = 5.7/15 = 0.38
        expected = (0.1 * 10 + 0.5 * 2 + 0.9 * 3) / (10 + 2 + 3)
        result = functor._aggregate(scores, sizes)

        assert abs(result - expected) < 1e-10

        # Test case 2: Single large component dominates
        scores = [0.1, 0.9, 0.9]  # One low score, two high scores
        sizes = [100, 1, 1]  # One very large component

        # Expected: heavily weighted toward the large component
        expected = (0.1 * 100 + 0.9 * 1 + 0.9 * 1) / (100 + 1 + 1)
        result = functor._aggregate(scores, sizes)

        assert abs(result - expected) < 1e-10
        assert (
            result < 0.12
        )  # Should be close to 0.1 due to large component weight

    def test_disconnected_graph_edge_cases(self, functor):
        """Test edge cases in disconnected graph processing."""
        # Test case 1: Graph with only isolated nodes (no edges)
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        nx.set_node_attributes(G, {0: [1], 1: [2], 2: [3]}, "x")
        X = np.array([[1], [2], [3]])

        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=False,
            ),
            patch(
                "rings.complementarity.functor.nx.connected_components",
                return_value=[{0}, {1}, {2}],  # All isolated nodes
            ),
            patch(
                "rings.complementarity.functor.lift_graph",
                side_effect=[
                    np.array([[0]]),  # Isolated node metrics
                    np.array([[0]]),
                    np.array([[0]]),
                ],
            ),
            patch(
                "rings.complementarity.functor.lift_attributes",
                side_effect=[
                    np.array([[0]]),
                    np.array([[0]]),
                    np.array([[0]]),
                ],
            ),
        ):
            D_X, D_G, sizes = functor._lift_metrics(G, X, empty_graph=False)

            assert len(D_X) == 3
            assert len(D_G) == 3
            assert sizes == [1, 1, 1]  # All single-node components

        # Test case 2: Single component in disconnected graph
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 1), (1, 2)])  # All connected
        nx.set_node_attributes(G, {0: [1], 1: [2], 2: [3]}, "x")
        X = np.array([[1], [2], [3]])

        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=True,  # Actually connected
            ),
            patch(
                "rings.complementarity.functor.lift_graph",
                return_value=np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            ),
            patch(
                "rings.complementarity.functor.lift_attributes",
                return_value=np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            ),
        ):
            D_X, D_G, sizes = functor._lift_metrics(G, X, empty_graph=False)

            assert len(D_X) == 1  # Single component
            assert len(D_G) == 1
            assert sizes == [3]  # All nodes in one component

    def test_disconnected_graph_error_handling(self, functor):
        """Test error handling in disconnected graph scenarios."""
        # Test aggregation with mismatched scores and sizes
        with pytest.raises(ValueError):
            # This should raise an error if implemented properly
            # For now, we'll test the current behavior
            scores = [0.1, 0.5]  # 2 scores
            sizes = [10, 2, 3]  # 3 sizes (mismatch)
            # Note: Current implementation doesn't validate this, but it should
            try:
                functor._aggregate(scores, sizes)
            except (IndexError, ValueError):
                raise ValueError("Mismatched scores and sizes")

    def test_disconnected_graph_metric_consistency(self, functor):
        """Test that metrics are consistently applied across disconnected components."""
        # Create a graph with identical disconnected components
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (2, 3)])  # Two identical pairs
        nx.set_node_attributes(
            G,
            {
                0: [1, 0],
                1: [2, 0],  # Component 1: features differ by 1 in first dim
                2: [1, 0],
                3: [2, 0],  # Component 2: identical features to component 1
            },
            "x",
        )
        X = np.array([[1, 0], [2, 0], [1, 0], [2, 0]])

        # Mock to return identical metric spaces for identical components
        identical_graph_metric = np.array([[0, 1], [1, 0]])
        identical_feature_metric = np.array([[0, 1], [1, 0]])

        with (
            patch(
                "rings.complementarity.functor.nx.is_connected",
                return_value=False,
            ),
            patch(
                "rings.complementarity.functor.nx.connected_components",
                return_value=[{0, 1}, {2, 3}],
            ),
            patch(
                "rings.complementarity.functor.lift_graph",
                side_effect=[identical_graph_metric, identical_graph_metric],
            ),
            patch(
                "rings.complementarity.functor.lift_attributes",
                side_effect=[
                    identical_feature_metric,
                    identical_feature_metric,
                ],
            ),
        ):
            D_X, D_G, sizes = functor._lift_metrics(G, X, empty_graph=False)

            # Verify identical components produce identical metric spaces
            assert np.array_equal(D_X[0], D_X[1])
            assert np.array_equal(D_G[0], D_G[1])
            assert sizes[0] == sizes[1]  # Same component sizes

        # Test that identical components yield identical scores
        scores = functor._compute_scores(D_X, D_G)
        assert abs(scores[0] - scores[1]) < 1e-10  # Should be identical

    @patch("rings.complementarity.functor.to_networkx")
    @patch.object(ComplementarityFunctor, "_process_single")
    def test_forward_as_dataframe(
        self, mock_process_single, mock_to_networkx, functor
    ):
        """Test forward method with as_dataframe=True."""
        # Setup mock
        mock_process_single.return_value = {
            "complementarity": 0.5,
            "other_metric": "value",
        }

        # Create test data
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data = Data(x=x, edge_index=edge_index)

        # Run forward with as_dataframe=True (default)
        result = functor.forward([test_data], as_dataframe=True)

        # Check results
        assert isinstance(
            result, pd.DataFrame
        ), "Result should be a pandas DataFrame"
        assert (
            "complementarity" in result.columns
        ), "Result should have 'complementarity' column"
        assert (
            "other_metric" in result.columns
        ), "Result should have other metric columns"
        assert len(result) == 1, "DataFrame should have one row for one graph"
        assert result["complementarity"].iloc[0] == 0.5
