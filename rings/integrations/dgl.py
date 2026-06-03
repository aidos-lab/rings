"""DGL compatibility wrappers for core RINGS perturbations.

This module keeps :mod:`rings.perturbations` as the source of truth by
converting ``dgl.DGLGraph`` objects to temporary PyG ``Data`` objects, applying
the perturbation, then converting back to DGL.
"""

from typing import Callable, Optional

import torch
from torch_geometric.data import Data

from rings.perturbations import (
    CompleteGraph,
    EmptyFeatures,
    EmptyGraph,
    Original,
    RandomFeatures,
    RandomGraph,
)

_DGL_IMPORT_ERROR = None
try:
    import dgl
except Exception as exc:  # pragma: no cover - optional dependency import surface
    dgl = None
    _DGL_IMPORT_ERROR = exc


def _check_dgl():
    if dgl is None:
        raise ImportError(
            "DGL is required for these perturbations. "
            "Install it with 'pip install dgl'."
        ) from _DGL_IMPORT_ERROR


def dgl_to_pyg(g: "dgl.DGLGraph", feat_name: str = "x") -> Data:
    """Convert a homogeneous DGL graph to a PyG ``Data`` object."""
    _check_dgl()
    src, dst = g.edges()
    data = Data(
        edge_index=torch.stack([src, dst], dim=0),
        num_nodes=g.num_nodes(),
    )

    for key, value in g.ndata.items():
        if key == feat_name:
            data.x = value.clone()
        else:
            data[f"ndata__{key}"] = value.clone()
    return data


def pyg_to_dgl(data: Data, feat_name: str = "x", device=None) -> "dgl.DGLGraph":
    """Convert a PyG ``Data`` object to a homogeneous DGL graph."""
    _check_dgl()
    if data.edge_index is None:
        src = torch.empty((0,), dtype=torch.int64)
        dst = torch.empty((0,), dtype=torch.int64)
    else:
        src, dst = data.edge_index[0], data.edge_index[1]

    if device is not None:
        src = src.to(device)
        dst = dst.to(device)

    new_g = dgl.graph((src, dst), num_nodes=data.num_nodes, device=device)
    if getattr(data, "x", None) is not None:
        x = data.x if device is None else data.x.to(device)
        new_g.ndata[feat_name] = x

    for key, value in data.to_dict().items():
        if not key.startswith("ndata__"):
            continue
        out_key = key.split("ndata__", 1)[1]
        new_g.ndata[out_key] = value if device is None else value.to(device)
    return new_g


def as_dgl_transform(
    pyg_transform: Callable, feat_name: str = "x"
) -> Callable[["dgl.DGLGraph"], "dgl.DGLGraph"]:
    """Wrap a PyG perturbation for DGL graphs via round-trip conversion."""

    def _transform(g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        _check_dgl()
        data = dgl_to_pyg(g, feat_name=feat_name)
        out = pyg_transform(data)
        return pyg_to_dgl(out, feat_name=feat_name, device=g.device)

    return _transform


class DGLOriginal:
    """DGL wrapper for :class:`rings.perturbations.Original`."""

    def __init__(self, feat_name: str = "x"):
        self._transform = as_dgl_transform(Original(), feat_name=feat_name)

    def __call__(self, g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        return self._transform(g)


class DGLEmptyFeatures:
    """DGL wrapper for :class:`rings.perturbations.EmptyFeatures`."""

    def __init__(self, feat_name: str = "x"):
        self.feat_name = feat_name
        self._transform = as_dgl_transform(EmptyFeatures(), feat_name=feat_name)

    def __call__(self, g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        return self._transform(g)


class DGLRandomFeatures:
    """DGL wrapper for :class:`rings.perturbations.RandomFeatures`."""

    def __init__(
        self,
        shuffle: bool = False,
        feat_name: str = "x",
        generator: Optional[torch.Generator] = None,
    ):
        self.shuffle = shuffle
        self.feat_name = feat_name
        self.generator = generator
        self._transform = RandomFeatures(shuffle=shuffle)
        if self.generator is not None:
            self._transform.generator = self.generator
        self._dgl_transform = as_dgl_transform(self._transform, feat_name=feat_name)

    def __call__(self, g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        return self._dgl_transform(g)


class DGLEmptyGraph:
    """DGL wrapper for :class:`rings.perturbations.EmptyGraph`."""

    def __init__(self, feat_name: str = "x"):
        self._transform = as_dgl_transform(EmptyGraph(), feat_name=feat_name)

    def __call__(self, g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        return self._transform(g)


class DGLCompleteGraph:
    """DGL wrapper for :class:`rings.perturbations.CompleteGraph`."""

    def __init__(self, feat_name: str = "x"):
        self._transform = as_dgl_transform(CompleteGraph(), feat_name=feat_name)

    def __call__(self, g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        return self._transform(g)


class DGLRandomGraph:
    """DGL wrapper for :class:`rings.perturbations.RandomGraph`."""

    def __init__(self, p: float = 0.1, generator: Optional[torch.Generator] = None):
        self._transform = RandomGraph(p=p)
        if generator is not None:
            self._transform.generator = generator
        self._dgl_transform = as_dgl_transform(self._transform)

    def __call__(self, g: "dgl.DGLGraph") -> "dgl.DGLGraph":
        return self._dgl_transform(g)
