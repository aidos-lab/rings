"""Utility Functions and Helper Transforms"""

import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform

#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper Transforms                                        |
#  ╰──────────────────────────────────────────────────────────╯


class Shuffle(BaseTransform):
    """
    A transform that shuffles node features and/or edges in a graph.

    This transform can randomize the relationship between node features and graph
    structure by shuffling feature vectors among nodes, shuffling edges between nodes,
    or both. It preserves the feature and edge distributions while destroying their
    correlations.

    Parameters
    ----------
    shuffle_edges : bool, default=False
        If True, shuffle the order of edges and reassign them to random nodes.
    shuffle_features : bool, default=False
        If True, shuffle the order of node features among nodes.
    generator : torch.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.utils import Shuffle
    >>>
    >>> # Create a simple graph with features and edges
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
    >>> edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    >>> data = Data(x=x, edge_index=edge_index)
    >>>
    >>> # Shuffle only features
    >>> transform1 = Shuffle(shuffle_features=True)
    >>> t1_data = transform1(data.clone())
    >>> # Feature vectors are preserved but reassigned among nodes
    >>> original_features = set(tuple(f.tolist()) for f in data.x)
    >>> shuffled_features = set(tuple(f.tolist()) for f in t1_data.x)
    >>> assert original_features == shuffled_features
    >>> assert t1_data.edge_index.equal(data.edge_index)
    >>>
    >>> # Shuffle only edges
    >>> transform2 = Shuffle(shuffle_edges=True)
    >>> t2_data = transform2(data.clone())
    >>> # Edge structure is changed, features unchanged
    >>> assert t2_data.x.equal(data.x)
    >>> assert not t2_data.edge_index.equal(data.edge_index)
    """

    def __init__(self, shuffle_edges=False, shuffle_features=False, generator=None):
        """
        Initialize the Shuffle transform.

        Parameters
        ----------
        shuffle_edges : bool, default=False
            If True, shuffle the order of edges and reassign them to random nodes.
        shuffle_features : bool, default=False
            If True, shuffle the order of node features among nodes.
        generator : torch.Generator, optional
            Random number generator for reproducibility. If None, a new generator is created.
        """
        self.shuffle_edges = shuffle_edges
        self.shuffle_features = shuffle_features
        self.generator = generator

    def forward(self, data):
        """
        Apply the shuffle transformation to the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Transformed data object with shuffled features and/or edges.
        """
        if self.shuffle_features:
            data = self._shuffle_features(data)
        if self.shuffle_edges:
            data = self._shuffle_edges(data)
        return data

    def _shuffle_features(self, data):
        """
        Shuffle node features among nodes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph with shuffled node features.

        Notes
        -----
        This method preserves the distribution of feature vectors but
        reassigns them randomly to different nodes, destroying any
        correlation between node features and graph structure.
        """
        if hasattr(data, "x") and data.x is not None:
            # Shuffle node features among nodes
            num_nodes = data.num_nodes
            perm = torch.randperm(
                num_nodes, device=data.x.device, generator=self.generator
            )
            data.x = data.x[perm]
        return data

    def _shuffle_edges(self, data):
        """
        Shuffle edges by randomizing their target nodes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph with shuffled edges.

        Notes
        -----
        This method preserves the source nodes and the total number of edges,
        but randomly reassigns the target node for each edge. Self-loops are
        prevented by reassigning edges that would create them.

        Raises
        ------
        ValueError
            If the data object doesn't have num_nodes defined.
        """
        if hasattr(data, "edge_index") and data.edge_index is not None:
            # Retrieve the source and target nodes from edge_index
            source_nodes = data.edge_index[0]
            target_nodes = data.edge_index[1]
            num_nodes = data.num_nodes
            if num_nodes is None:
                raise ValueError("Data object must have `num_nodes` defined.")

            # Create a random permutation of node indices for the target nodes
            perm = torch.randperm(
                num_nodes,
                device=data.edge_index.device,
                generator=self.generator,
            )

            # Shuffle target nodes using the permutation
            shuffled_target_nodes = perm[target_nodes]

            # Ensure the original number of edges is preserved
            source_nodes, shuffled_target_nodes = ensure_no_self_loops(
                source_nodes,
                shuffled_target_nodes,
                num_nodes,
                generator=self.generator,
            )
            assert shuffled_target_nodes.size() == target_nodes.size()
            # Update edge_index with the shuffled edges
            data.edge_index = torch.stack([source_nodes, shuffled_target_nodes], dim=0)

        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Utility Functions                                        |
#  ╰──────────────────────────────────────────────────────────╯


def is_connected(data):
    """
    Check if a graph is connected.

    A connected graph has a path between every pair of nodes. This function converts
    the PyTorch Geometric graph to NetworkX format and uses NetworkX's is_connected
    function to check connectivity.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data object.

    Returns
    -------
    bool
        True if the graph is connected, False otherwise.
        Empty graphs (0 nodes) are considered connected by convention.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.utils import is_connected
    >>>
    >>> # Create a connected graph: 0-1-2
    >>> edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    >>> data1 = Data(edge_index=edge_index1, num_nodes=3)
    >>> assert is_connected(data1) == True
    >>>
    >>> # Create a disconnected graph: 0-1  2
    >>> edge_index2 = torch.tensor([[0], [1]], dtype=torch.long)
    >>> data2 = Data(edge_index=edge_index2, num_nodes=3)
    >>> assert is_connected(data2) == False
    >>>
    >>> # Empty graph
    >>> data3 = Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=0)
    >>> assert is_connected(data3) == True  # Connected by convention
    """
    if data.num_nodes == 0:
        # Empty graphs are connected by convention
        return True

    G = to_networkx(data, to_undirected=True)

    return nx.is_connected(G)


def ensure_no_self_loops(source_nodes, target_nodes, num_nodes, generator):
    """
    Ensure no self-loops exist in the shuffled edges.

    If any self-loops are detected after shuffling, they are replaced with random
    edges (by assigning a random target node) to maintain the original number of edges.

    Parameters
    ----------
    source_nodes : torch.Tensor
        Source nodes of the edges.
    target_nodes : torch.Tensor
        Target nodes of the edges.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple of torch.Tensor
        Updated (source_nodes, target_nodes) without self-loops.

    Notes
    -----
    This method iteratively replaces target nodes that would create self-loops
    until all self-loops are eliminated. The source nodes are kept intact.
    """
    # Identify self-loops
    self_loop_mask = source_nodes == target_nodes
    num_self_loops = self_loop_mask.sum().item()
    if num_self_loops > 0:
        random_target_nodes = torch.randint(
            0,
            num_nodes,
            (num_self_loops,),
            device=target_nodes.device,
            generator=generator,
        )

        # Generate new target nodes ensuring they're not equal to the source nodes to avoid creating self-loops
        valid_targets_mask = random_target_nodes != source_nodes[self_loop_mask]
        while not valid_targets_mask.all():
            invalid_indices = ~valid_targets_mask
            random_target_nodes[invalid_indices] = torch.randint(
                0,
                num_nodes,
                (invalid_indices.sum().item(),),
                device=target_nodes.device,
                generator=generator,
            )
            valid_targets_mask = random_target_nodes != source_nodes[self_loop_mask]

        # Replace only target nodes to keep the source nodes intact
        target_nodes[self_loop_mask] = random_target_nodes

    return source_nodes, target_nodes
