import torch
from torch_geometric.data import Data

from rings.perturbations import (
    Original,
    EmptyFeatures,
    CompleteFeatures,
    RandomFeatures,
    EmptyGraph,
    CompleteGraph,
    RandomGraph,
    RandomConnectedGraph,
)
from rings.utils import is_connected


class TestPerturbations:
    def setup_method(self):
        """Set up test data before each test method."""
        # Create a simple graph for testing with 4 nodes
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        # Features with dimension 3
        self.x = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float
        )
        # Create a simple data object
        self.data = Data(x=self.x, edge_index=self.edge_index, num_nodes=4)

        # Additional test data with different properties
        # Create a path graph 0-1-2-3
        self.path_edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        self.path_data = Data(x=self.x, edge_index=self.path_edge_index, num_nodes=4)

        # Set fixed seed for reproducibility
        torch.manual_seed(42)

    def test_original_transform(self):
        """Test that Original transform doesn't modify the data."""
        transform = Original()
        transformed_data = transform(self.data.clone())

        # Original transform should not alter any data attributes
        assert torch.equal(transformed_data.x, self.data.x)
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)
        assert transformed_data.num_nodes == self.data.num_nodes

    def test_empty_features_transform(self):
        """Test EmptyFeatures transform assigns zero vectors."""
        transform = EmptyFeatures()
        transformed_data = transform(self.data.clone())

        # Check that feature dimensions are correct
        assert transformed_data.x.shape == (self.data.num_nodes, 1)

        # Check that all features are zero
        assert torch.all(transformed_data.x == 0)

        # Check that graph structure remains unchanged
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)

    def test_complete_features_transform(self):
        """Test CompleteFeatures transform assigns one-hot vectors."""
        max_nodes = 10
        transform = CompleteFeatures(max_nodes=max_nodes)
        transformed_data = transform(self.data.clone())

        # Check feature dimensions
        assert transformed_data.x.shape == (self.data.num_nodes, max_nodes)

        # Verify one-hot encoding is correct
        for i in range(self.data.num_nodes):
            # Each node gets a one-hot vector matching its index
            expected = torch.zeros(max_nodes)
            expected[i] = 1
            assert torch.equal(transformed_data.x[i], expected)

        # Check that graph structure remains unchanged
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)

    def test_random_features_transform_randn(self):
        """Test RandomFeatures transform with random normal sampling."""
        # Test with default parameters (randn)
        transform = RandomFeatures(shuffle=False)
        transformed_data = transform(self.data.clone())

        # Check feature dimensions remain the same
        assert transformed_data.x.shape == self.data.x.shape

        # Check that features were changed
        assert not torch.equal(transformed_data.x, self.data.x)

        # Check that graph structure remains unchanged
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)

    def test_random_features_transform_fixed_dimension(self):
        """Test RandomFeatures transform with fixed dimension."""
        fixed_dim = 10
        transform = RandomFeatures(shuffle=False, fixed_dimension=fixed_dim)
        transformed_data = transform(self.data.clone())

        # Check that feature dimensions were changed to the fixed dimension
        assert transformed_data.x.shape == (self.data.num_nodes, fixed_dim)

        # Check that graph structure remains unchanged
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)

    def test_random_features_transform_shuffle(self):
        """Test RandomFeatures transform with shuffle=True."""
        transform = RandomFeatures(shuffle=True)
        transformed_data = transform(self.data.clone())

        # Check that dimensions are preserved
        assert transformed_data.x.shape == self.data.x.shape

        # Features should be shuffled, not newly generated
        assert not torch.equal(transformed_data.x, self.data.x)

        # Check that the feature vectors are preserved (just reordered)
        original_features = set(tuple(x.tolist()) for x in self.data.x)
        shuffled_features = set(tuple(x.tolist()) for x in transformed_data.x)
        assert original_features == shuffled_features

        # Check that graph structure remains unchanged
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)

    def test_empty_graph_transform(self):
        """Test EmptyGraph transform removes all edges."""
        transform = EmptyGraph()
        transformed_data = transform(self.data.clone())

        # Check that there are no edges
        assert transformed_data.edge_index.shape == (2, 0)

        # Check that node features remain unchanged
        assert torch.equal(transformed_data.x, self.data.x)

        # Check that edge attributes are removed
        assert transformed_data.edge_attr is None

    def test_complete_graph_transform(self):
        """Test CompleteGraph transform creates a fully connected graph."""
        transform = CompleteGraph()
        transformed_data = transform(self.data.clone())

        # Number of edges in a complete graph with n nodes = n*(n-1)
        n = self.data.num_nodes
        expected_num_edges = n * (
            n - 1
        )  # Direction matters, so double directional edges

        # Check the edge count is correct
        assert transformed_data.edge_index.shape[1] == expected_num_edges

        # Check that every possible edge exists
        for i in range(n):
            for j in range(n):
                if i != j:  # No self-loops
                    # Find if edge i->j exists in the edge_index
                    edge_exists = (
                        (
                            (transformed_data.edge_index[0] == i)
                            & (transformed_data.edge_index[1] == j)
                        )
                        .any()
                        .item()
                    )
                    assert edge_exists, f"Edge from {i} to {j} should exist"

        # Check that node features remain unchanged
        assert torch.equal(transformed_data.x, self.data.x)

        # Check that edge attributes are removed
        assert transformed_data.edge_attr is None

    def test_random_graph_transform_shuffle(self):
        """Test RandomGraph transform with shuffle=True."""
        transform = RandomGraph(shuffle=True)
        transformed_data = transform(self.data.clone())

        # Check that the number of edges remains the same
        assert transformed_data.edge_index.shape[1] == self.data.edge_index.shape[1]

        # Check that the edges were actually shuffled
        assert not torch.equal(transformed_data.edge_index, self.data.edge_index)

        # Check that node features remain unchanged
        assert torch.equal(transformed_data.x, self.data.x)

    def test_random_graph_transform_p_value(self):
        """Test RandomGraph transform with specific edge probability."""
        # Test with p=0.5
        p = 0.5
        transform = RandomGraph(p=p)
        transformed_data = transform(self.data.clone())

        # Check that node features remain unchanged
        assert torch.equal(transformed_data.x, self.data.x)

        # Check that edge attributes are removed
        assert transformed_data.edge_attr is None

    def test_random_graph_transform_documentation_example(self):
        """Test RandomGraph transform using the example from documentation."""
        # Create a simple graph as shown in documentation
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=4)

        # Example from documentation: Random graph with same number of edges
        torch.manual_seed(42)  # For reproducibility
        transform = RandomGraph()
        transformed_data = transform(data.clone())

        # Should have same number of edges but different structure
        assert transformed_data.edge_index.shape[1] == data.edge_index.shape[1]
        assert not torch.equal(transformed_data.edge_index, data.edge_index)

        # Verify the transform maintains other properties
        assert transformed_data.num_nodes == data.num_nodes
        assert transformed_data.edge_attr is None

    def test_random_connected_graph_transform_shuffle(self):
        """Test RandomConnectedGraph transform with shuffle=True."""
        transform = RandomConnectedGraph(shuffle=True)
        transformed_data = transform(self.data.clone())

        # Check that the number of edges remains the same
        assert transformed_data.edge_index.shape[1] == self.data.edge_index.shape[1]

        # Check that the edges were actually shuffled
        assert not torch.equal(transformed_data.edge_index, self.data.edge_index)

        # Check that node features remain unchanged
        assert torch.equal(transformed_data.x, self.data.x)

        # Check that the graph is connected
        assert is_connected(transformed_data)

    def test_random_connected_graph_transform_p_value(self):
        """Test RandomConnectedGraph transform with specific edge probability."""
        # Test with p=0.5
        p = 0.5
        transform = RandomConnectedGraph(p=p)
        transformed_data = transform(self.data.clone())

        # Expected number of edges with p=0.5 but need at least n-1 for connected
        n = self.data.num_nodes
        min_edges_for_connected = n - 1  # Minimum edges needed for a connected graph

        # Check that we have at least the minimum required edges
        assert transformed_data.edge_index.shape[1] >= min_edges_for_connected

        # Check that node features remain unchanged
        assert torch.equal(transformed_data.x, self.data.x)

        # Check that the graph is connected
        assert is_connected(transformed_data)

        # Check that edge attributes are removed
        assert transformed_data.edge_attr is None

    def test_connectivity_guarantees(self):
        """Test that RandomConnectedGraph always produces connected graphs."""
        # Create a test case with a disconnected graph
        disconnected_edge_index = torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,  # Only nodes 0 and 1 are connected
        )
        disconnected_data = Data(
            x=self.x, edge_index=disconnected_edge_index, num_nodes=4
        )

        # Apply the RandomConnectedGraph transform
        transform = RandomConnectedGraph(p=0.3)
        result = transform(disconnected_data.clone())

        # Verify the result is connected
        assert is_connected(result)

        # Test multiple times to ensure consistency
        for _ in range(5):
            transform = RandomConnectedGraph(p=0.1)  # Low probability to test edge case
            result = transform(disconnected_data.clone())
            assert is_connected(result)

    def test_random_connected_graph_spanning_tree(self):
        """Test that RandomConnectedGraph creates at minimum a spanning tree."""
        # Test with very low probability to ensure we get close to minimal connected graph
        p = 0.01
        transform = RandomConnectedGraph(p=p)
        transformed_data = transform(self.data.clone())

        # A connected graph with n nodes must have at least n-1 edges
        n = self.data.num_nodes
        min_edges = n - 1

        assert transformed_data.edge_index.shape[1] >= min_edges

        # Verify it's connected
        assert is_connected(transformed_data)
