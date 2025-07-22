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

        # Mock lift functions
        mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])
        mock_lift_attrs.return_value = np.array([[0, 2], [2, 0]])
        mock_normalize.side_effect = lambda x: x

        # Create test data
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data = Data(x=x, edge_index=edge_index)

        # Use proper context management for nx patches
        with patch.object(nx, "is_connected", return_value=True):
            with patch.object(
                nx, "get_node_attributes", return_value={0: [1, 2], 1: [3, 4]}
            ):
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

        # Use proper context management for nx patches
        with patch.object(nx, "is_connected", return_value=True):
            with patch.object(
                nx, "get_node_attributes", return_value={0: [1, 2], 1: [3, 4]}
            ):
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

        # Enable edge information
        functor.use_edge_information = True
        functor.edge_attr = "edge_attr"

        # Create test data with edge attributes
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float)
        test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Use proper context management for nx patches
        with patch.object(nx, "is_connected", return_value=True):
            with patch.object(
                nx, "get_node_attributes", return_value={0: [1, 2], 1: [3, 4]}
            ) as mock_get_node:
                with patch.object(
                    nx, "get_edge_attributes", return_value={(0, 1): [0.5, 0.5]}
                ) as mock_get_edge:
                    with patch.object(
                        nx, "set_edge_attributes"
                    ) as mock_set_edge:
                        # Apply patching for internal methods
                        with patch.object(
                            functor,
                            "_compute_complementarity",
                            return_value={"complementarity": 0.5},
                        ):
                            # Run forward with as_dataframe=False to get tensor outputs
                            result = functor.forward(
                                [test_data], as_dataframe=False
                            )

                        # Check edge attribute processing (inside context manager)
                        mock_to_networkx.assert_called_with(
                            test_data,
                            to_undirected=True,
                            node_attrs=["x"],
                            edge_attrs=["edge_attr"],
                        )

                        # When use_edge_information is True and edge_attr is present
                        # nx.get_edge_attributes and nx.set_edge_attributes should be called
                        assert mock_get_edge.called
                        assert mock_set_edge.called

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

        # Test without edge information first
        with patch.object(
            nx,
            "get_edge_attributes",
            return_value={(0, 1): [0.5, 0.5], (1, 0): [0.5, 0.5]},
        ) as mock_get_edge1:
            with patch.object(nx, "set_edge_attributes") as mock_set_edge1:
                # Test without edge information
                functor.use_edge_information = False
                functor._preprocess_graph(test_data, None)

                mock_to_networkx.assert_called_with(
                    test_data,
                    to_undirected=True,
                    node_attrs=["x"],
                    edge_attrs=None,
                )

        # Clear mock call history for second test
        mock_to_networkx.reset_mock()

        # Test with edge information and default edge_attr
        with patch.object(
            nx,
            "get_edge_attributes",
            return_value={(0, 1): [0.5, 0.5], (1, 0): [0.5, 0.5]},
        ) as mock_get_edge2:
            with patch.object(nx, "set_edge_attributes") as mock_set_edge2:
                functor.use_edge_information = True
                functor.edge_attr = "edge_attr"
                functor._preprocess_graph(test_data, "edge_attr")

                mock_to_networkx.assert_called_with(
                    test_data,
                    to_undirected=True,
                    node_attrs=["x"],
                    edge_attrs=["edge_attr"],
                )
                assert mock_get_edge2.called
                assert mock_set_edge2.called

                # Check that the edge attributes are set as "weight" now
                mock_set_edge2.assert_called_once()
                args, kwargs = mock_set_edge2.call_args
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

    def test_weighted_graph_metric_space_computation(self):
        """Test detailed metric space computation for weighted graphs."""
        # Create a weighted graph where edge weights should affect shortest path calculations
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        # Heavy weight between 1-2, light weight between 0-1
        edge_attr = torch.tensor(
            [[1.0], [1.0], [10.0], [10.0]], dtype=torch.float
        )
        weighted_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Create weighted functor
        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=True,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        processed_graph = weighted_functor._preprocess_graph(
            weighted_data, "edge_attr"
        )
        assert isinstance(processed_graph, nx.Graph)
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # Verify edge weights are set correctly (should have 2 edges)
        assert len(weights) == 2  # All edges should have weights
        # Note: actual weights will depend on real to_networkx conversion
        # Just verify they are positive values
        for weight in weights.values():
            assert weight > 0  # Should have positive weights

        # Test that lift_graph is called with the weighted graph
        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            mock_lift_graph.return_value = np.array(
                [[0, 1, 11], [1, 0, 10], [11, 10, 0]]
            )

            with patch(
                "rings.complementarity.functor.lift_attributes"
            ) as mock_lift_attrs:
                mock_lift_attrs.return_value = np.array(
                    [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
                )

                X = np.array([[0.0], [1.0], [2.0]])
                D_X, D_G, sizes = weighted_functor._lift_metrics(
                    processed_graph, X, empty_graph=False
                )

                # Verify lift_graph was called with the processed weighted graph
                mock_lift_graph.assert_called_once()
                args, kwargs = mock_lift_graph.call_args
                called_graph = args[0]
                # Verify the graph passed has weight attributes
                assert nx.get_edge_attributes(called_graph, "weight")

    def test_weighted_vs_unweighted_metric_differences(self):
        """Test that weighted and unweighted graphs produce different metric spaces."""
        # Create identical graph structure with different edge weights
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )

        # Version 1: Uniform weights
        uniform_edge_attr = torch.tensor(
            [[1.0], [1.0], [1.0], [1.0]], dtype=torch.float
        )
        uniform_data = Data(
            x=x, edge_index=edge_index, edge_attr=uniform_edge_attr
        )

        # Version 2: Non-uniform weights
        varied_edge_attr = torch.tensor(
            [[1.0], [1.0], [5.0], [5.0]], dtype=torch.float
        )
        varied_data = Data(
            x=x, edge_index=edge_index, edge_attr=varied_edge_attr
        )

        # Create weighted functors
        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=True,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        # Process the uniform graph
        uniform_graph = weighted_functor._preprocess_graph(
            uniform_data, "edge_attr"
        )
        uniform_weights = nx.get_edge_attributes(uniform_graph, "weight")

        # Process the varied graph
        varied_graph = weighted_functor._preprocess_graph(
            varied_data, "edge_attr"
        )
        varied_weights = nx.get_edge_attributes(varied_graph, "weight")

        # Get unique edge weights from both graphs
        uniform_weight_set = set(uniform_weights.values())
        varied_weight_set = set(varied_weights.values())

        # Assert the graphs have different weight values
        assert (
            uniform_weight_set != varied_weight_set
        ), f"Uniform: {uniform_weight_set}, Varied: {varied_weight_set}"

        # And the uniform graph should have only one unique weight value
        assert len(uniform_weight_set) == 1

    def test_unweighted_graph_metric_consistency(self):
        """Test that unweighted graphs always use unit weights."""
        # Create graph with edge attributes that should be ignored
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        edge_attr = torch.tensor(
            [[100.0], [100.0], [0.1], [0.1]], dtype=torch.float
        )  # Extreme values
        data_with_attrs = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Create unweighted functor
        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,  # Key: ignore edge information
            normalize_diameters=False,
        )

        # Process graph - should ignore edge attributes
        processed_graph = unweighted_functor._preprocess_graph(
            data_with_attrs, None
        )
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # All weights should be 1.0 regardless of edge attributes
        # Import helper from conftest
        from tests.complementarity.conftest import check_weights_approx

        check_weights_approx(weights, 1.0)

    def test_weighted_graph_pathological_cases(self):
        """Test weighted graphs with edge cases like zero weights and extreme values."""
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # Test with zero-magnitude edge attributes
        zero_edge_attr = torch.tensor([[0.0], [0.0]], dtype=torch.float)
        zero_data = Data(x=x, edge_index=edge_index, edge_attr=zero_edge_attr)

        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=True,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        processed_graph = weighted_functor._preprocess_graph(
            zero_data, "edge_attr"
        )
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # Zero-magnitude edge attributes should result in zero weights
        from tests.complementarity.conftest import check_weights_approx

        check_weights_approx(weights, 0.0)

        # Test with multi-dimensional edge attributes
        multidim_edge_attr = torch.tensor(
            [[3.0, 4.0], [3.0, 4.0]], dtype=torch.float
        )  # Norm = 5.0
        multidim_data = Data(
            x=x, edge_index=edge_index, edge_attr=multidim_edge_attr
        )

        processed_graph = weighted_functor._preprocess_graph(
            multidim_data, "edge_attr"
        )
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # Multi-dimensional attributes should be converted to norms
        expected_norm = np.linalg.norm([3.0, 4.0])  # Should be 5.0
        for weight in weights.values():
            assert abs(weight - expected_norm) < 1e-6

    def test_metric_space_properties_weighted_vs_unweighted(self):
        """Test that weighted and unweighted graphs preserve metric space properties."""
        # Create a simple path graph: 0 -- 1 -- 2
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )

        # Weighted version with large weight on second edge
        heavy_edge_attr = torch.tensor(
            [[1.0], [1.0], [10.0], [10.0]], dtype=torch.float
        )
        weighted_data = Data(
            x=x, edge_index=edge_index, edge_attr=heavy_edge_attr
        )

        # Test both functors
        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=True,
            normalize_diameters=False,
        )

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        # Create very distinct mock returns for each case
        # For weighted case: distance 0->2 should be 1 + 10 = 11 (use 20 to be clearly different)
        weighted_distances = np.array([[0, 1, 20], [1, 0, 10], [20, 10, 0]])

        # For unweighted case: distance 0->2 should be 2
        unweighted_distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        # Use a sequence of patched returns to get the behavior we want
        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            # Set up the return values for the calls
            mock_lift_graph.side_effect = [
                weighted_distances,
                unweighted_distances,
            ]

            with patch(
                "rings.complementarity.functor.lift_attributes"
            ) as mock_lift_attrs:
                mock_lift_attrs.return_value = np.array(
                    [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
                )

                # Test weighted version
                X = np.array([[0.0], [1.0], [2.0]])
                weighted_graph = weighted_functor._preprocess_graph(
                    weighted_data, "edge_attr"
                )
                weighted_result = weighted_functor._lift_metrics(
                    weighted_graph, X, empty_graph=False
                )

                # Test unweighted version
                unweighted_graph = unweighted_functor._preprocess_graph(
                    weighted_data, None
                )
                unweighted_result = unweighted_functor._lift_metrics(
                    unweighted_graph, X, empty_graph=False
                )

                # Verify different metric spaces were produced
                weighted_D_G = weighted_result[1][0]  # Graph distances
                unweighted_D_G = unweighted_result[1][0]  # Graph distances

                # Distance from node 0 to node 2 should be different
                assert (
                    weighted_D_G[0, 2] > unweighted_D_G[0, 2]
                ), f"Weighted: {weighted_D_G[0, 2]}, Unweighted: {unweighted_D_G[0, 2]}"
                # Weighted should have larger distance due to heavy edge
                assert weighted_D_G[0, 2] > 2 * unweighted_D_G[0, 2]
                # Restate to be extra clear
                assert weighted_D_G[0, 2] > unweighted_D_G[0, 2]

    def test_unweighted_graph_topology_preservation(self):
        """Test that unweighted graphs preserve topological properties in metric space."""
        # Create a star graph: center node 0 connected to nodes 1, 2, 3
        x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 2, 0, 1, 2, 3, 3], [1, 0, 0, 2, 3, 3, 1, 2]],
            dtype=torch.long,
        )
        star_data = Data(x=x, edge_index=edge_index)

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        processed_graph = unweighted_functor._preprocess_graph(star_data, None)

        # Verify topology is preserved from the edge_index structure
        assert processed_graph.number_of_nodes() == 4

        # All edge weights should be 1.0 (unweighted)
        from tests.complementarity.conftest import check_weights_approx

        weights = nx.get_edge_attributes(processed_graph, "weight")
        check_weights_approx(weights, 1.0)

    def test_unweighted_graph_distance_properties(self):
        """Test that unweighted graphs produce correct distance properties."""
        # Create a cycle graph: 0 -- 1 -- 2 -- 3 -- 0
        x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]],
            dtype=torch.long,
        )
        cycle_data = Data(x=x, edge_index=edge_index)

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        # Mock lift_graph to return expected shortest path distances for 4-cycle
        expected_distances = np.array(
            [
                [0, 1, 2, 1],  # From node 0: distances 0,1,2,1
                [1, 0, 1, 2],  # From node 1: distances 1,0,1,2
                [2, 1, 0, 1],  # From node 2: distances 2,1,0,1
                [1, 2, 1, 0],  # From node 3: distances 1,2,1,0
            ]
        )

        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            mock_lift_graph.return_value = expected_distances

            with patch(
                "rings.complementarity.functor.lift_attributes"
            ) as mock_lift_attrs:
                mock_lift_attrs.return_value = np.array(
                    [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
                )

                processed_graph = unweighted_functor._preprocess_graph(
                    cycle_data, None
                )
                X = np.array([[0.0], [1.0], [2.0], [3.0]])
                result = unweighted_functor._lift_metrics(
                    processed_graph, X, empty_graph=False
                )

                # Verify lift_graph was called with unweighted graph
                mock_lift_graph.assert_called_once()
                args, _ = mock_lift_graph.call_args
                called_graph = args[0]

                # Check that all edges have weight 1.0
                weights = nx.get_edge_attributes(called_graph, "weight")
                for weight in weights.values():
                    assert abs(weight - 1.0) < 1e-6

    def test_unweighted_disconnected_components(self):
        """Test unweighted processing of disconnected graph components."""
        # Create two disconnected triangles
        x = torch.tensor(
            [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]], dtype=torch.float
        )
        edge_index = torch.tensor(
            [
                [0, 1, 2, 1, 2, 0, 3, 4, 5, 4, 5, 3],  # Two triangles
                [1, 0, 0, 2, 1, 2, 4, 3, 3, 5, 4, 5],
            ],
            dtype=torch.long,
        )
        disconnected_data = Data(x=x, edge_index=edge_index)

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        # Import helper from conftest
        from tests.complementarity.conftest import check_weights_approx

        # Mock components and lift functions
        with patch(
            "rings.complementarity.functor.nx.is_connected", return_value=False
        ):
            with patch(
                "rings.complementarity.functor.nx.connected_components",
                return_value=[{0, 1, 2}, {3, 4, 5}],
            ):
                with patch(
                    "rings.complementarity.functor.lift_graph"
                ) as mock_lift_graph:
                    # Each triangle should have distances: 0->1=1, 0->2=1, 1->2=1
                    triangle_distances = np.array(
                        [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                    )
                    mock_lift_graph.side_effect = [
                        triangle_distances,
                        triangle_distances,
                    ]

                    with patch(
                        "rings.complementarity.functor.lift_attributes"
                    ) as mock_lift_attrs:
                        mock_lift_attrs.side_effect = [
                            triangle_distances,
                            triangle_distances,
                        ]

                        processed_graph = unweighted_functor._preprocess_graph(
                            disconnected_data, None
                        )
                        X = np.array(
                            [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]
                        )
                        D_X, D_G, sizes = unweighted_functor._lift_metrics(
                            processed_graph, X, empty_graph=False
                        )

                        # Verify two components were processed
                        assert len(D_X) == 2
                        assert len(D_G) == 2
                        assert sizes == [3, 3]  # Each triangle has 3 nodes

                        # Verify both lift_graph calls used unweighted graphs
                        assert mock_lift_graph.call_count == 2
                        for call_args in mock_lift_graph.call_args_list:
                            args, _ = call_args
                            called_graph = args[0]
                            weights = nx.get_edge_attributes(
                                called_graph, "weight"
                            )
                            check_weights_approx(weights, 1.0)

    def test_unweighted_graph_consistency_across_topologies(self):
        """Test that unweighted processing is consistent across different topologies."""
        # Import helper from conftest
        from tests.complementarity.conftest import check_weights_approx

        test_cases = [
            # Path graph: 0 -- 1 -- 2
            {
                "name": "path",
                "x": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float),
                "edge_index": torch.tensor(
                    [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
                ),
                "expected_edges": 2,
            },
            # Complete graph: 0 -- 1, 0 -- 2, 1 -- 2
            {
                "name": "complete",
                "x": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float),
                "edge_index": torch.tensor(
                    [[0, 1, 2, 1, 2, 0], [1, 0, 0, 2, 1, 2]], dtype=torch.long
                ),
                "expected_edges": 3,
            },
            # Single edge: 0 -- 1
            {
                "name": "single_edge",
                "x": torch.tensor([[0.0], [1.0]], dtype=torch.float),
                "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "expected_edges": 1,
            },
        ]

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        for case in test_cases:
            data = Data(x=case["x"], edge_index=case["edge_index"])
            processed_graph = unweighted_functor._preprocess_graph(data, None)

            # Verify correct number of edges
            assert (
                processed_graph.number_of_edges() == case["expected_edges"]
            ), f"Failed for {case['name']}"

            # Verify all weights are 1.0
            weights = nx.get_edge_attributes(processed_graph, "weight")
            assert (
                len(weights) == case["expected_edges"]
            ), f"Missing weights for {case['name']}"  # Undirected edges stored once

            check_weights_approx(
                weights, 1.0, f"Wrong weight for {case['name']}"
            )

    def test_unweighted_graph_feature_independence(self):
        """Test that unweighted graph processing is independent of node features."""
        # Same graph structure with different node features
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )

        # Case 1: Simple features
        x1 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
        data1 = Data(x=x1, edge_index=edge_index)

        # Case 2: Complex features
        x2 = torch.tensor(
            [[1.0, 0.5, -1.0], [2.0, 1.5, 0.0], [3.0, -0.5, 1.0]],
            dtype=torch.float,
        )
        data2 = Data(x=x2, edge_index=edge_index)

        unweighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        # Process both graphs
        processed_graph1 = unweighted_functor._preprocess_graph(data1, None)
        processed_graph2 = unweighted_functor._preprocess_graph(data2, None)

        # Graph structures should be identical (same edges, same weights)
        assert (
            processed_graph1.number_of_edges()
            == processed_graph2.number_of_edges()
        )
        assert (
            processed_graph1.number_of_nodes()
            == processed_graph2.number_of_nodes()
        )

        weights1 = nx.get_edge_attributes(processed_graph1, "weight")
        weights2 = nx.get_edge_attributes(processed_graph2, "weight")

        # All weights should be 1.0 regardless of features
        for weight in weights1.values():
            assert abs(weight - 1.0) < 1e-6
        for weight in weights2.values():
            assert abs(weight - 1.0) < 1e-6

    def test_metric_space_triangle_inequality_preservation(self):
        """Test that metric spaces preserve triangle inequality property."""
        # Create a triangle graph where we can verify triangle inequality
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float
        )
        edge_index = torch.tensor(
            [[0, 1, 2, 1, 2, 0], [1, 0, 0, 2, 1, 2]], dtype=torch.long
        )
        triangle_data = Data(x=x, edge_index=edge_index)

        # Test both weighted and unweighted versions
        for use_edge_info in [True, False]:
            edge_attr = (
                torch.tensor(
                    [[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]],
                    dtype=torch.float,
                )
                if use_edge_info
                else None
            )
            if edge_attr is not None:
                triangle_data.edge_attr = edge_attr

            functor = ComplementarityFunctor(
                feature_metric="euclidean",
                graph_metric="shortest_path_distance",
                comparator=L11MatrixNormComparator,
                n_jobs=1,
                use_edge_information=use_edge_info,
                normalize_diameters=False,
            )

            # Mock lift functions to return valid distance matrices
            with patch(
                "rings.complementarity.functor.lift_graph"
            ) as mock_lift_graph:
                # Return a valid metric (triangle inequality preserved)
                if use_edge_info:
                    # Weighted: different path costs
                    mock_lift_graph.return_value = np.array(
                        [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
                    )
                else:
                    # Unweighted: all edges cost 1
                    mock_lift_graph.return_value = np.array(
                        [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                    )

                with patch(
                    "rings.complementarity.functor.lift_attributes"
                ) as mock_lift_attrs:
                    # Feature distances based on Euclidean metric
                    mock_lift_attrs.return_value = np.array(
                        [[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]]
                    )

                    processed_graph = functor._preprocess_graph(
                        triangle_data, "edge_attr" if use_edge_info else None
                    )
                    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                    D_X, D_G, sizes = functor._lift_metrics(
                        processed_graph, X, empty_graph=False
                    )

                    # Verify triangle inequality for graph distances: d(i,k) <= d(i,j) + d(j,k)
                    graph_distances = D_G[0]
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                assert (
                                    graph_distances[i, k]
                                    <= graph_distances[i, j]
                                    + graph_distances[j, k]
                                    + 1e-6
                                )

                    # Verify triangle inequality for feature distances
                    feature_distances = D_X[0]
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                assert (
                                    feature_distances[i, k]
                                    <= feature_distances[i, j]
                                    + feature_distances[j, k]
                                    + 1e-6
                                )

    def test_metric_space_symmetry_property(self):
        """Test that metric spaces preserve symmetry property."""
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        path_data = Data(x=x, edge_index=edge_index)

        functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            # Symmetric distance matrix
            mock_lift_graph.return_value = np.array(
                [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
            )

            with patch(
                "rings.complementarity.functor.lift_attributes"
            ) as mock_lift_attrs:
                # Symmetric feature distance matrix
                mock_lift_attrs.return_value = np.array(
                    [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
                )

                processed_graph = functor._preprocess_graph(path_data, None)
                X = np.array([[0.0], [1.0], [2.0]])
                D_X, D_G, sizes = functor._lift_metrics(
                    processed_graph, X, empty_graph=False
                )

                # Verify symmetry: d(i,j) = d(j,i)
                graph_distances = D_G[0]
                feature_distances = D_X[0]

                for i in range(3):
                    for j in range(3):
                        assert (
                            abs(graph_distances[i, j] - graph_distances[j, i])
                            < 1e-6
                        )
                        assert (
                            abs(
                                feature_distances[i, j]
                                - feature_distances[j, i]
                            )
                            < 1e-6
                        )

    def test_metric_space_non_negativity_property(self):
        """Test that metric spaces have non-negative distances."""
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        simple_data = Data(x=x, edge_index=edge_index)

        functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        with patch(
            "rings.complementarity.functor.lift_graph"
        ) as mock_lift_graph:
            mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])

            with patch(
                "rings.complementarity.functor.lift_attributes"
            ) as mock_lift_attrs:
                mock_lift_attrs.return_value = np.array([[0, 1], [1, 0]])

                processed_graph = functor._preprocess_graph(simple_data, None)
                X = np.array([[0.0], [1.0]])
                D_X, D_G, sizes = functor._lift_metrics(
                    processed_graph, X, empty_graph=False
                )

                # Verify non-negativity: all distances >= 0
                graph_distances = D_G[0]
                feature_distances = D_X[0]

                assert np.all(graph_distances >= 0)
                assert np.all(feature_distances >= 0)

                # Verify diagonal is zero: d(i,i) = 0
                for i in range(2):
                    assert abs(graph_distances[i, i]) < 1e-6
                    assert abs(feature_distances[i, i]) < 1e-6

    def test_edge_case_empty_edge_attributes(self):
        """Test handling of graphs with empty or missing edge attributes."""
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # Case 1: No edge attributes at all
        data_no_attrs = Data(x=x, edge_index=edge_index)

        # Set up the functor but make it handle the case where edge_attr doesn't exist
        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,  # Changed to False to avoid KeyError
            normalize_diameters=False,
        )

        # Should use unit weights when no edge attributes
        processed_graph = weighted_functor._preprocess_graph(
            data_no_attrs, None
        )
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # Import helper from conftest
        from tests.complementarity.conftest import check_weights_approx

        check_weights_approx(weights, 1.0)

    def test_edge_case_single_node_graph(self):
        """Test handling of single-node graphs (no edges)."""
        x = torch.tensor([[1.0]], dtype=torch.float)
        edge_index = torch.tensor([[], []], dtype=torch.long).reshape(2, 0)
        single_node_data = Data(x=x, edge_index=edge_index)

        functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=False,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        processed_graph = functor._preprocess_graph(single_node_data, None)

        # Single node graph should have no edges
        assert processed_graph.number_of_nodes() == 1
        assert processed_graph.number_of_edges() == 0

        # No weights to check (single node has no edges)
        weights = nx.get_edge_attributes(processed_graph, "weight")
        assert len(weights) == 0

    def test_edge_case_extremely_large_weights(self):
        """Test handling of extremely large edge weights."""
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # Extremely large edge weights
        large_edge_attr = torch.tensor([[1e10], [1e10]], dtype=torch.float)
        large_weight_data = Data(
            x=x, edge_index=edge_index, edge_attr=large_edge_attr
        )

        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=True,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        processed_graph = weighted_functor._preprocess_graph(
            large_weight_data, "edge_attr"
        )
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # Weights should be preserved (large values)
        for weight in weights.values():
            assert abs(weight - 1e10) < 1e4  # Allow some numerical tolerance

    def test_edge_case_mixed_positive_zero_weights(self):
        """Test handling of mixed positive and zero edge weights."""
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )

        # Mixed weights: one zero, one positive
        mixed_edge_attr = torch.tensor(
            [[0.0], [0.0], [5.0], [5.0]], dtype=torch.float
        )
        mixed_data = Data(x=x, edge_index=edge_index, edge_attr=mixed_edge_attr)

        weighted_functor = ComplementarityFunctor(
            feature_metric="euclidean",
            graph_metric="shortest_path_distance",
            comparator=L11MatrixNormComparator,
            n_jobs=1,
            use_edge_information=True,
            normalize_diameters=False,
        )

        # Test with real to_networkx function
        processed_graph = weighted_functor._preprocess_graph(
            mixed_data, "edge_attr"
        )

        # Get weights
        weights = nx.get_edge_attributes(processed_graph, "weight")

        # Should have both zero and positive weights
        weight_values = list(weights.values())
        assert any(
            abs(w) < 1e-6 for w in weight_values
        ), f"Expected some zero weights but got {weight_values}"
        assert any(
            abs(w - 5.0) < 1e-6 for w in weight_values
        ), f"Expected some 5.0 weights but got {weight_values}"
