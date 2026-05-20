"""This file defines the Mode Perturbations, i.e Graph Transformations, of the RINGS Framework."""

import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dense_to_sparse

from rings.utils import Shuffle, is_connected, ensure_no_self_loops


class Original(BaseTransform):
    """
    A placeholder transform that returns the input node features & graph data without modifications.

    This transform serves as a baseline for comparing other transforms' effects on graphs.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> from rings.perturbations import Original
    >>> dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    >>> data = dataset[0]  # Get the first graph
    >>> transform = Original()
    >>> transformed_data = transform(data)
    >>> # The transformed data is identical to the original data
    >>> assert transformed_data == data
    """

    def forward(self, data):
        """
        Return the original, unmodified data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Unmodified graph data object.
        """
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Perturb Node Features                                    |
#  ╰──────────────────────────────────────────────────────────╯


class EmptyFeatures(BaseTransform):
    """
    A transform that assigns identical features (zero vector) to each node.

    This transform removes all node feature information by replacing existing features
    with identical zero vectors, preserving only graph structure information.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> from rings.perturbations import EmptyFeatures
    >>> dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
    >>> data = dataset[0]  # Get the first graph
    >>> transform = EmptyFeatures()
    >>> transformed_data = transform(data)
    >>> # Check that all node features are zero vectors
    >>> import torch
    >>> assert torch.all(transformed_data.x == 0)
    >>> # Check that all node features have dimension 1
    >>> assert transformed_data.x.size(1) == 1
    """

    def forward(self, data):
        """
        Assign zero vectors as features to each node.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph data object with zero node features.
        """
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, 1), dtype=torch.float)
        return data


class CompleteFeatures(BaseTransform):
    """
    A transform that assigns unique node IDs as features to each node.

    Each node is represented by a padded one-hot encoded vector, creating maximally
    distinctive node features where each node can be uniquely identified.

    Parameters
    ----------
    max_nodes : int
        Maximum number of nodes for one-hot feature encoding. This determines
        the dimension of the one-hot vectors.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.perturbations import CompleteFeatures
    >>> # Create a graph with 3 nodes
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    >>> data = Data(edge_index=edge_index, num_nodes=3)
    >>> # Transform with max_nodes=5
    >>> transform = CompleteFeatures(max_nodes=5)
    >>> transformed_data = transform(data)
    >>> print(transformed_data.x)
    tensor([[1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.]])
    """

    def __init__(self, max_nodes):
        """
        Initialize the CompleteFeatures transform.

        Parameters
        ----------
        max_nodes : int
            Maximum number of nodes for one-hot feature encoding.
        """
        self.max_nodes = max_nodes

    def forward(self, data):
        """
        Apply the transform to assign one-hot encoded node IDs as features.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph data object with one-hot node features.
        """
        num_nodes = data.num_nodes  # Total number of nodes in the graph

        # Create a one-hot encoding for each node
        one_hot = torch.eye(self.max_nodes, dtype=torch.float)

        # Select the rows corresponding to the actual nodes
        data.x = one_hot[:num_nodes]

        return data


class RandomFeatures(BaseTransform):
    """
    A transform that randomizes node features.

    This transform either samples new features from a standard normal distribution or
    shuffles existing node features between nodes, effectively destroying any meaningful
    correlation between node features and graph structure while preserving the feature
    distribution.

    Parameters
    ----------
    shuffle : bool, default=False
        If True, shuffle existing node features among nodes.
        If False, replace features with random values from a standard normal distribution.
    fixed_dimension : int, optional
        Fixed dimension for new random features. If None, use the original feature dimension.
        Only used when shuffle=False.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.perturbations import RandomFeatures
    >>> # Create a simple graph with features
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
    >>> edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    >>> data = Data(x=x, edge_index=edge_index)
    >>>
    >>> # Example 1: Random normal features
    >>> transform1 = RandomFeatures(shuffle=False)
    >>> t1_data = transform1(data.clone())
    >>> # Features will be different, but dimensions preserved
    >>> assert t1_data.x.shape == data.x.shape
    >>> assert not torch.allclose(t1_data.x, data.x)
    >>>
    >>> # Example 2: Shuffled features
    >>> transform2 = RandomFeatures(shuffle=True)
    >>> t2_data = transform2(data.clone())
    >>> # Original features should be present but in different order
    >>> original_set = {tuple(row.tolist()) for row in data.x}
    >>> shuffled_set = {tuple(row.tolist()) for row in t2_data.x}
    >>> assert original_set == shuffled_set
    >>>
    >>> # Example 3: Random features with fixed dimension
    >>> transform3 = RandomFeatures(fixed_dimension=5)
    >>> t3_data = transform3(data.clone())
    >>> assert t3_data.x.shape == (3, 5)  # 3 nodes, 5 features
    """

    def __init__(self, shuffle=False, fixed_dimension=None):
        """
        Initialize the RandomFeatures transform.

        Parameters
        ----------
        shuffle : bool, default=False
            If True, shuffle existing node features among nodes.
            If False, replace features with random values from a standard normal distribution.
        fixed_dimension : int, optional
            Fixed dimension for new random features. If None, use the original feature dimension.
            Only used when shuffle=False.
        """
        self.dimension = fixed_dimension
        self.shuffle = shuffle

        current_seed = torch.initial_seed()
        self.generator = torch.Generator().manual_seed(current_seed)

    def forward(self, data):
        """
        Apply the transform to assign or shuffle node features.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph data object with modified features.
        """
        if self.shuffle:
            return self._shuffle(data)
        else:
            return self._randomize_features(data, dimension=self.dimension)

    def _shuffle(self, data):
        """
        Shuffle node features among nodes in the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with shuffled features.
        """
        return Shuffle(shuffle_features=True, generator=self.generator)(data)

    def _randomize_features(self, data, dimension=None):
        """
        Sample random node features from a standard normal distribution using `torch.randn`.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data object.
        dimension : int, optional
            Fixed feature dimension (if None, use original dimension).

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with randomized features.
        """
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.size(0)
            dim = data.x.size(1) if dimension is None else dimension
            data.x = torch.randn(num_nodes, dim, generator=self.generator)
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Perturb Graph Structure                                  |
#  ╰──────────────────────────────────────────────────────────╯


class EmptyGraph(BaseTransform):
    """
    A transform that removes all edges from the graph, creating an empty graph.

    This transform preserves node features but removes all edges, thereby eliminating
    any graph structure information. The resulting graph consists of isolated nodes.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.perturbations import EmptyGraph
    >>> # Create a simple graph with features and edges
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
    >>> edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    >>> data = Data(x=x, edge_index=edge_index)
    >>>
    >>> # Apply transform
    >>> transform = EmptyGraph()
    >>> transformed_data = transform(data)
    >>>
    >>> # Check that all edges are removed
    >>> assert transformed_data.edge_index.shape == (2, 0)
    >>> # Check that node features remain unchanged
    >>> assert torch.equal(transformed_data.x, data.x)
    """

    def forward(self, data):
        """
        Remove all edges from the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph data object with no edges.
        """
        data.edge_index = torch.tensor(
            [[], []], dtype=torch.long
        )  # Remove edges by setting edge_index to an empty tensor
        data.edge_attr = None  # Optionally remove edge attributes if they exist

        return data


class CompleteGraph(BaseTransform):
    """
    A transform that replaces the existing graph structure with a complete graph.

    This transform preserves node features but connects every pair of nodes with an edge,
    creating a fully connected graph where each node is directly connected to all other nodes.
    Self-loops are excluded.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.perturbations import CompleteGraph
    >>> # Create a simple graph with 3 nodes
    >>> edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # 0-1-2 path
    >>> data = Data(edge_index=edge_index, num_nodes=3)
    >>>
    >>> # Apply transform
    >>> transform = CompleteGraph()
    >>> transformed_data = transform(data)
    >>>
    >>> # In a complete graph with 3 nodes, there should be 6 directed edges (3×2)
    >>> assert transformed_data.edge_index.shape[1] == 6
    >>>
    >>> # Check that every possible edge (except self-loops) exists
    >>> edges = set(zip(transformed_data.edge_index[0].tolist(),
    ...                 transformed_data.edge_index[1].tolist()))
    >>> expected_edges = {(0,1), (1,0), (0,2), (2,0), (1,2), (2,1)}
    >>> assert edges == expected_edges
    """

    def forward(self, data):
        """
        Convert the graph into a complete graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph data object as a complete graph.
        """
        num_nodes = data.num_nodes  # Number of nodes in the graph

        # Create a complete graph adjacency matrix (all nodes connected to each other)
        adj = torch.ones((num_nodes, num_nodes), dtype=torch.long) - torch.eye(
            num_nodes, dtype=torch.long
        )

        # Convert dense adjacency matrix to edge_index format
        edge_index, _ = dense_to_sparse(adj)

        # Assign the complete graph to data.edge_index
        data.edge_index = edge_index

        # Remove any existing edge attributes
        data.edge_attr = None

        return data


class RandomGraph(BaseTransform):
    """
    A transform that replaces the existing graph structure with a random graph.

    The graph is generated either using an Erdos-Renyi model (with probability p) or
    by randomly shuffling the current edges. Node features are preserved while graph
    structure is randomized.

    Parameters
    ----------
    p : float, optional
        Probability of an edge existing between any two nodes in the Erdos-Renyi model.
        If None, the same number of edges as in the original graph is used.
    shuffle : bool, default=False
        If True, shuffle the existing graph structure instead of creating a new one.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.perturbations import RandomGraph
    >>> # Create a simple graph
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    >>> data = Data(edge_index=edge_index, num_nodes=4)
    >>>
    >>> # Example 1: Random graph with same number of edges
    >>> torch.manual_seed(42)  # For reproducibility
    >>> transform1 = RandomGraph()
    >>> t1_data = transform1(data.clone())
    >>> # Should have same number of edges but different structure
    >>> assert t1_data.edge_index.shape[1] == data.edge_index.shape[1]
    >>> assert not torch.equal(t1_data.edge_index, data.edge_index)
    >>>
    >>> # Example 2: Random graph with specified edge probability
    >>> transform2 = RandomGraph(p=0.5)
    >>> t2_data = transform2(data.clone())
    >>> # Expected edges with p=0.5: 0.5 * 4 * 3 / 2 = 3 (before removing self-loops)
    >>>
    >>> # Example 3: Shuffle existing edges
    >>> transform3 = RandomGraph(shuffle=True)
    >>> t3_data = transform3(data.clone())
    >>> # Should have same number of edges
    >>> assert t3_data.edge_index.shape[1] == data.edge_index.shape[1]
    """

    def __init__(self, p=None, shuffle=False):
        """
        Initialize the RandomGraph transform.

        Parameters
        ----------
        p : float, optional
            Probability of an edge existing between any two nodes in the Erdos-Renyi model.
            If None, the same number of edges as in the original graph is used.
        shuffle : bool, default=False
            If True, shuffle the existing graph structure instead of creating a new one.
        """
        self.p = p
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(torch.initial_seed())

    def forward(self, data):
        """
        Replace the graph structure with a random graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph data object with random graph structure.
        """
        if self.shuffle:
            return self._shuffle(data)
        else:
            return self._randomize_graph(data)

    def _shuffle(self, data):
        """
        Shuffle edges in the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph with shuffled edges.
        """
        return Shuffle(shuffle_edges=True, generator=self.generator)(data)

    def _randomize_graph(self, data):
        """
        Generate a random Erdos-Renyi graph structure.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph with random structure.
        """
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Data object must have `num_nodes` defined.")

        num_edges = self._set_num_edges(data, num_nodes, self.p)

        row = torch.randint(
            0, num_nodes, (num_edges,), device="cpu", generator=self.generator
        )
        col = torch.randint(
            0, num_nodes, (num_edges,), device="cpu", generator=self.generator
        )

        # Ensure no self-loops
        row, col = ensure_no_self_loops(row, col, num_nodes, generator=self.generator)

        # Create sparse adjacency matrix
        edge_index = torch.stack([row, col], dim=0).to(data.edge_index.device)
        data.edge_index = edge_index
        data.edge_attr = None

        return data

    @staticmethod
    def _set_num_edges(data, N, p):
        """
        Compute the number of edges for the random graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.
        N : int
            Number of nodes in the graph.
        p : float, optional
            Edge probability.

        Returns
        -------
        int
            Number of edges.

        Notes
        -----
        If p is None, the same number of edges as in the original graph is used.
        Otherwise, the number of edges is computed as p * N * (N-1) / 2.
        """
        num_edges = data.edge_index.size(1) if p is None else int(p * N * (N - 1) / 2)
        return num_edges


class RandomConnectedGraph(BaseTransform):
    """
    A transform that replaces the existing graph structure with a random connected graph.

    The graph is generated either by randomly shuffling the current edges or using a modified
    Erdos-Renyi model ensuring connectivity. The resulting graph is guaranteed to be connected,
    meaning there exists a path between any two nodes.

    Parameters
    ----------
    p : float, optional
        Edge probability parameter. If None, the same number of edges as in the original
        graph is used. The actual probability is adjusted to ensure connectivity.
    shuffle : bool, default=False
        If True, shuffle edges of the existing graph instead of creating a new structure.
        Multiple shuffle attempts may be performed until a connected graph is achieved.

    Examples
    --------
    >>> import torch
    >>> import networkx as nx
    >>> from torch_geometric.data import Data
    >>> from torch_geometric.utils import to_networkx
    >>> from rings.perturbations import RandomConnectedGraph
    >>>
    >>> # Create a simple graph
    >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    >>> data = Data(edge_index=edge_index, num_nodes=4)
    >>>
    >>> # Apply transform
    >>> torch.manual_seed(42)  # For reproducibility
    >>> transform = RandomConnectedGraph(p=0.5)
    >>> transformed_data = transform(data.clone())
    >>>
    >>> # Convert to networkx to check connectivity
    >>> G = to_networkx(transformed_data, to_undirected=True)
    >>> assert nx.is_connected(G)
    >>>
    >>> # With shuffle=True
    >>> transform2 = RandomConnectedGraph(shuffle=True)
    >>> t2_data = transform2(data.clone())
    >>> G2 = to_networkx(t2_data, to_undirected=True)
    >>> assert nx.is_connected(G2)
    """

    def __init__(self, p=None, shuffle=False):
        """
        Initialize the RandomConnectedGraph transform.

        Parameters
        ----------
        p : float, optional
            Edge probability parameter. If None, the same number of edges as in the
            original graph is used.
        shuffle : bool, default=False
            If True, shuffle edges instead of creating a new structure.
        """
        self.p = p
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(torch.initial_seed())

    def forward(self, data):
        """
        Generate a random connected graph structure.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph with connected structure.
        """
        transform = self._shuffle if self.shuffle else self._randomize_connected_graph
        data = transform(data)
        while not is_connected(data):
            data = transform(data)
        return data

    def _shuffle(self, data):
        """
        Shuffle edges in the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph with shuffled edges.
        """
        return Shuffle(shuffle_edges=True, generator=self.generator)(data)

    def _randomize_connected_graph(self, data):
        """
        Generate a random connected graph structure.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed graph with connected structure.
        """
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Data object must have `num_nodes` defined.")

        # Add additional random edges based on probability p
        edges = self._random_spanning_tree(num_nodes)
        edge_set = set(edges)

        num_additional_edges = self._set_num_additional_edges(
            data, num_nodes, self.p, len(edges)
        )

        while len(edge_set) < len(edges) + num_additional_edges:
            # Randomly sample a new edge
            u = torch.randint(
                0, num_nodes, (1,), device="cpu", generator=self.generator
            ).item()
            v = torch.randint(
                0, num_nodes, (1,), device="cpu", generator=self.generator
            ).item()

            # Avoid self-loops and duplicate edges
            if u != v:
                edge_set.add((min(u, v), max(u, v)))  # Use sorted edges for consistency

        # Convert edge_set to a PyTorch edge_index on CPU
        edge_index = torch.tensor(list(edge_set), dtype=torch.long, device="cpu").t()

        # Transfer the edge_index to the same device as the input data
        data.edge_index = edge_index.to(data.edge_index.device)
        data.edge_attr = None  # Clear edge attributes if they exist
        return data

    def _random_spanning_tree(self, N):
        """
        Generate a random spanning tree.

        Parameters
        ----------
        N : int
            Number of nodes in the graph.

        Returns
        -------
        list
            List of edges in the spanning tree.

        Notes
        -----
        A spanning tree is a minimal connected subgraph that includes all nodes.
        This implementation builds a tree iteratively by connecting each new node
        to a random previously added node.
        """
        # Generate a random spanning tree
        spanning_tree = []
        for i in range(1, N):
            # Connect each node to a random previous node
            spanning_tree.append(
                (i, torch.randint(0, i, (1,), generator=self.generator).item())
            )
        return spanning_tree

    @staticmethod
    def _set_num_additional_edges(data, N, p, l_tree):
        """
        Calculate the number of additional edges to add to a spanning tree.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.
        N : int
            Number of nodes in the graph.
        p : float, optional
            Edge probability parameter.
        l_tree : int
            Number of edges in the spanning tree.

        Returns
        -------
        int
            Number of additional edges to add.

        Notes
        -----
        This method calculates how many additional edges should be added
        after a spanning tree is created to reach the desired edge count.
        If p is None, it tries to match the original graph's edge count.
        Otherwise, it uses p to determine the total number of edges.
        """
        num_edges = data.edge_index.size(1) if p is None else int(p * N * (N - 1) / 2)
        num_new_edges = num_edges - l_tree
        return num_new_edges if num_new_edges > 0 else 0
