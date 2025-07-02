"""Core Functor for facilitating Mode Complementarity computation."""

from rings.complementarity.metrics import lift_attributes
from rings.complementarity.metrics import lift_graph
from rings.complementarity.utils import maybe_normalize_diameter

from torch_geometric.utils import to_networkx

import numpy as np
import networkx as nx
import pandas as pd

import joblib
import operator
import torch
import warnings
from typing import Dict, List, Union, Optional, Any


class ComplementarityFunctor(torch.nn.Module):
    """
    A functor for computing complementarity between graph structure and node features.

    This class computes complementarity by comparing the metric spaces derived from
    graph structure and node features using a specified comparator. It quantifies
    how well node features align with the graph structure, with lower values indicating
    stronger alignment.

    For disconnected graphs, the functor processes each connected component separately
    and aggregates the results using a weighted average, where component sizes serve
    as weights. This approach ensures that larger components have proportionally more
    influence on the final complementarity score.

    Parameters
    ----------
    feature_metric : callable
        The metric function used to compute distances between node features.
    graph_metric : callable
        The metric function used to compute distances in the graph structure.
    comparator : class
        A comparator class that implements a __call__ method to compare metric spaces.
    n_jobs : int, default=1
        Number of jobs to run in parallel. If 1, no parallelism is used.
    use_edge_information : bool, default=False
        Whether to use edge attributes in graph metric computation.
    normalize_diameters : bool, default=True
        Whether to normalize the diameters of metric spaces before comparison.
    **kwargs : dict
        Additional arguments passed to the comparator and metric functions.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from rings.complementarity.comparator import MatrixNormComparator
    >>> from rings.complementarity.metrics import standard_feature_metrics, shortest_path_distance
    >>>
    >>> # Create a simple graph where node features correspond to positions in the graph
    >>> # A path graph: 0 -- 1 -- 2 -- 3 with features encoding their positions
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    >>> # Node features encode positions: node 0 is at position 0, node 1 at position 1, etc.
    >>> x = torch.tensor([[0], [1], [2], [3]], dtype=torch.float)
    >>> data = Data(x=x, edge_index=edge_index)
    >>>
    >>> # Create functor with simple metrics
    >>> functor = ComplementarityFunctor(
    ...     feature_metric='euclidean',  # Use Euclidean distance for features
    ...     graph_metric='shortest_path_distance',  # Use shortest path for graph
    ...     comparator=MatrixNormComparator,  # Compare using matrix norm
    ...     n_jobs=1,
    ...     normalize_diameters=True,  # Normalize distances for fair comparison
    ... )
    >>>
    >>> # Compute complementarity for the graph
    >>> result = functor([data])
    >>> print(f"Complementarity score: {result['complementarity'].item():.4f}")
    Complementarity score: 0.0000
    >>>
    >>> # The score is 0, indicating perfect alignment between features and structure
    >>> # (node features perfectly correspond to their positions in the path)
    >>>
    >>> # Now create a graph with misaligned features
    >>> x_misaligned = torch.tensor([[3], [1], [2], [0]], dtype=torch.float)  # Swap 0 and 3
    >>> data_misaligned = Data(x=x_misaligned, edge_index=edge_index)
    >>>
    >>> # Compute complementarity for the misaligned graph
    >>> result = functor([data_misaligned])
    >>> print(f"Complementarity score: {result['complementarity'].item():.4f}")
    Complementarity score: 0.5000
    >>>
    >>> # The score is higher, indicating weaker alignment between features and structure
    >>> # (node features no longer match their positions in the path)
    """

    def __init__(
        self,
        feature_metric,
        graph_metric,
        comparator,
        n_jobs: int = 1,
        use_edge_information: bool = False,
        normalize_diameters: bool = True,
        edge_attr: Optional[str] = None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)

        self.n_jobs = n_jobs
        self.feature_metric = feature_metric
        self.graph_metric = graph_metric
        self.normalize_diameters = normalize_diameters
        self.use_edge_information = use_edge_information
        self.edge_attr = None

        # Build kwargs for metrics and comparator
        self.kwargs = kwargs.copy()

        # If no attribute convention then try PyG convention of "edge_attr"
        if self.use_edge_information:
            self.edge_attr = edge_attr if edge_attr is not None else "edge_attr"

        # Set up the comparator
        self.comparator = comparator(n_jobs=n_jobs, **self.kwargs)

    def forward(
        self,
        batch,
        as_dataframe: bool = True,
    ) -> Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
        """
        Compute complementarity for a batch of graphs in PyTorch Geometric format.

        Parameters
        ----------
        batch : list
            A batch of graphs in PyTorch Geometric format.
        as_dataframe : bool, default=True
            If True, return results as a pandas DataFrame, otherwise as a dictionary
            with tensor values or a list of dictionaries.

        Returns
        -------
        Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
            Results of complementarity computation for each graph in the batch.
            If as_dataframe=True, returns a pandas DataFrame.
            Otherwise, returns either a dictionary with tensor values or a list of dictionaries.

        Examples
        --------
        >>> from rings.complementarity.comparator import MatrixNormComparator
        >>> import torch_geometric.datasets as datasets
        >>> dataset = datasets.TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        >>> functor = ComplementarityFunctor(
        ...     feature_metric=lambda x, y: np.linalg.norm(x - y),
        ...     graph_metric=lambda x, y: abs(x - y),
        ...     comparator=MatrixNormComparator,
        ...     n_jobs=4,
        ...     normalize_diameters=True
        ... )
        >>> # Get first 5 graphs as a batch
        >>> batch = [dataset[i] for i in range(5)]
        >>> results = functor.forward(batch)
        >>> print(results)
        """
        batch_size = len(batch)

        # Process batch in parallel or sequentially
        if self.n_jobs > 1 and batch_size > 1:
            outputs = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._process_single)(batch[i])
                for i in range(batch_size)
            )
        else:
            outputs = [
                self._process_single(batch[i]) for i in range(batch_size)
            ]

        # Convert to DataFrame if requested
        if as_dataframe:
            return pd.DataFrame(outputs)

        # Otherwise, return as tensor dictionary or list
        complementarity_scores = torch.as_tensor(
            list(map(operator.itemgetter("complementarity"), outputs)),
            dtype=torch.float,
        )

        result = {"complementarity": complementarity_scores}

        if len(outputs) > 0:
            other_keys = list(outputs[0].keys())
            if "complementarity" in other_keys:
                other_keys.remove("complementarity")

            for key in other_keys:
                result[key] = [output[key] for output in outputs]

        return result

    def _process_single(self, data) -> Dict[str, Any]:
        """
        Process a single graph for complementarity calculation.

        Parameters
        ----------
        data : torch_geometric.Data
            The graph in PyTorch Geometric format.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the complementarity score and any additional metrics.
        """
        # Step 1: Preprocess graph
        G = self._preprocess_graph(data, self.edge_attr)

        # Step 2: Compute complementarity
        return self._compute_complementarity(G)

    def _preprocess_graph(self, data, edge_attr=None) -> nx.Graph:
        """
        Preprocess a graph from PyTorch Geometric format to NetworkX.

        In particular, this function creates a weighted graph (if requested by the user). When given multidimensional edge attributes, we construct scalar edge weights by applying a norm.

        We assign these to an attribute called "weight" in the NetworkX graph, which then will be picked up by default when computing graph metrics.


        Parameters
        ----------
        data : torch_geometric.Data
            The graph in PyTorch Geometric format.

        Returns
        -------
        networkx.Graph
            The preprocessed graph in NetworkX format.
        """
        # Only allow string or None for edge_attr
        if not (isinstance(edge_attr, str) or edge_attr is None):
            edge_attr = None

        G = to_networkx(
            data,
            to_undirected=True,
            node_attrs=["x"] if "x" in data else None,
            edge_attrs=[edge_attr] if edge_attr is not None else edge_attr,
        )

        # If using edge information and edge_attr is present, set weights from edge_attr
        if (
            self.use_edge_information
            and isinstance(edge_attr, str)
            and edge_attr in data
        ):
            attributes = nx.get_edge_attributes(G, edge_attr)
            # Convert multi-dimensional edge attributes to scalars using norm
            attributes = {
                edge: np.linalg.norm(attribute)
                for edge, attribute in attributes.items()
            }
            nx.set_edge_attributes(G, attributes, "weight")
        else:
            # Always set all edge weights to 1.0 if not using edge information
            nx.set_edge_attributes(
                G, {edge: 1.0 for edge in G.edges()}, "weight"
            )

        return G

    def _compute_complementarity(
        self, G, return_metric_spaces: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate complementarity of a single graph.

        This method breaks down the complementarity calculation into clear steps:
        1. Extract node features
        2. Lift graph structure and node features to metric spaces
        3. Normalize metric spaces if required
        4. Compute complementarity scores
        5. Aggregate results from multiple components if needed

        Parameters
        ----------
        G : networkx.Graph
            The graph, represented as `networkx.Graph` with optional
            node attributes and edge attributes. Node attributes are
            assumed to be stored in an attribute called `x`, whereas
            edge attributes are optionally stored under `edge_attr`.

        return_metric_spaces : bool, default=False
            Return the metric spaces being compared.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complementarity score and any additional metrics.
        """
        # Extract node features
        X = np.asarray(list(nx.get_node_attributes(G, "x").values()))

        # Validate input: treat empty graph or empty feature matrix the same
        empty_graph = G.number_of_edges() == 0
        if len(X) == 0:
            warnings.warn("Feature matrix X is empty, skipping graph.")
            return self.invalid_data

        # Handle connected components
        metric_spaces = self._lift_metrics(G, X, empty_graph)
        if metric_spaces is None:
            return self.invalid_data

        D_X, D_G, sizes = metric_spaces

        # Normalize metric spaces if required
        if self.normalize_diameters:
            D_X, D_G = self._normalize_metrics(D_X, D_G)

        # Compute scores for each component
        scores = self._compute_scores(D_X, D_G)

        # Aggregate results
        weighted_average_score = self._aggregate(scores, sizes)

        # Prepare return dictionary
        result = {"complementarity": weighted_average_score}

        if return_metric_spaces:
            result["D_X"] = D_X
            result["D_G"] = D_G

        return result

    def _lift_metrics(self, G, X, empty_graph: bool):
        """
        Lift graph structure and node features to metric spaces.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        X : numpy.ndarray
            Node feature matrix.
        empty_graph : bool
            Whether the graph has no edges.

        Returns
        -------
        tuple or None
            A tuple of (D_X, D_G, sizes) containing feature metric spaces,
            graph metric spaces, and component sizes. Returns None if operation fails.
        """
        # Check if the graph is connected
        if not nx.is_connected(G) and not empty_graph:
            # Process each connected component separately
            components = list(nx.connected_components(G))

            # Lift graphs for each component
            D_G = [
                lift_graph(
                    G.subgraph(C), metric=self.graph_metric, **self.kwargs
                )
                for C in components
            ]

            # Lift attributes (node features) for each component
            D_X = [
                lift_attributes(
                    X[list(C)],
                    metric=self.feature_metric,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                )
                for C in components
            ]

            sizes = [len(C) for C in components]
        else:
            # Lift Graph into Metric Space
            D_G = [lift_graph(G, metric=self.graph_metric, **self.kwargs)]

            # Lift Features into Metric Space
            D_X = [
                lift_attributes(
                    X,
                    metric=self.feature_metric,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                )
            ]

            sizes = [len(G.nodes())]

        return D_X, D_G, sizes

    def _normalize_metrics(self, D_X, D_G):
        """
        Normalize the diameters of metric spaces.

        Parameters
        ----------
        D_X : list of numpy.ndarray
            List of feature metric spaces.
        D_G : list of numpy.ndarray
            List of graph metric spaces.

        Returns
        -------
        tuple
            A tuple of normalized (D_X, D_G).
        """
        # Normalize diameters for both D_X and D_G
        D_X_normalized = [maybe_normalize_diameter(d_x) for d_x in D_X]
        D_G_normalized = [maybe_normalize_diameter(d_g) for d_g in D_G]

        return D_X_normalized, D_G_normalized

    def _compute_scores(self, D_X, D_G):
        """
        Compute complementarity scores for each component.

        Parameters
        ----------
        D_X : list of numpy.ndarray
            List of feature metric spaces.
        D_G : list of numpy.ndarray
            List of graph metric spaces.

        Returns
        -------
        list
            List of complementarity scores for each component.
        """
        # Compute complementarity scores for each component
        return [
            self.comparator(d_x, d_g)["score"] for d_x, d_g in zip(D_X, D_G)
        ]

    def _aggregate(self, scores, sizes):
        """
        Aggregate scores from multiple components using weighted average.

        For disconnected graphs, this method computes a weighted average of
        complementarity scores across all connected components. Each component's
        score is weighted by its size (number of nodes), ensuring that larger
        components have proportionally more influence on the final score.

        This weighted averaging approach is chosen because:
        1. **Representativeness**: Larger components contain more structural and
           feature information, so their complementarity patterns are more
           statistically significant.
        2. **Stability**: Small components (especially isolated nodes) can have
           extreme complementarity values that don't reflect the overall graph
           structure-feature relationship.
        3. **Interpretability**: The weighted average provides a graph-level
           complementarity score that reflects the dominant structural patterns.

        Parameters
        ----------
        scores : list
            List of complementarity scores for each connected component.
        sizes : list
            List of component sizes (number of nodes) used as weights.

        Returns
        -------
        float
            Weighted average complementarity score across all components.

        Notes
        -----
        If all component sizes are zero (edge case), falls back to simple
        arithmetic mean with a warning.

        Raises
        ------
        ValueError
            If the lengths of scores and sizes lists don't match.
        """
        # Ensure scores and sizes have the same length
        if len(scores) != len(sizes):
            raise ValueError("Scores and sizes must have the same length")

        # Compute the weighted average score
        if sum(sizes) > 0:
            return np.average(scores, weights=sizes)
        else:
            # If sizes sum to zero, use simple average or return NaN
            warnings.warn("Weights sum to zero, using simple average instead")
            if scores:
                return np.mean(scores)
            else:
                return np.nan

    @property
    def invalid_data(self):
        """
        Return a dictionary with NaN score and other standardized fields.

        Returns
        -------
        dict
            Dictionary with keys:
            - score: NaN (invalid data)
            - pvalue: None
            - pvalue_adjusted: None
            - significant: None
            - method: The norm used
        """
        return {
            "complementarity": np.nan,
        }
