from collections import Counter

import pytest
import torch

try:
    import dgl
except Exception as exc:
    pytest.skip(f"DGL unavailable in test runtime: {exc}", allow_module_level=True)

from rings.integrations import (  # noqa: E402
    DGLCompleteGraph,
    DGLEmptyFeatures,
    DGLEmptyGraph,
    DGLOriginal,
    DGLRandomFeatures,
    DGLRandomGraph,
    SeparabilityStudy,
)
from rings.integrations.dgl import as_dgl_transform  # noqa: E402
from rings.perturbations import EmptyFeatures  # noqa: E402


def _citation_like_graph():
    # Directed "citation-like" toy graph:
    # paper 0 cites 1 and 2; paper 3 cites 0 and 2; plus reciprocal links in a cluster.
    src = torch.tensor([0, 0, 1, 2, 3, 3, 4, 5], dtype=torch.int64)
    dst = torch.tensor([1, 2, 2, 1, 0, 2, 5, 4], dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=6)
    g.ndata["x"] = torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [0.5, 1.1, 0.3],
            [0.7, 0.9, 0.4],
            [1.3, 0.1, 0.8],
            [0.2, 0.3, 1.2],
            [0.1, 0.4, 1.1],
        ],
        dtype=torch.float32,
    )
    return g


class TestDGLIntegration:
    def test_original_matches_identity_semantics(self):
        g = _citation_like_graph()
        out = DGLOriginal()(g)
        assert out is not g
        assert torch.equal(out.ndata["x"], g.ndata["x"])
        out_src, out_dst = out.edges()
        src, dst = g.edges()
        assert torch.equal(out_src, src)
        assert torch.equal(out_dst, dst)

    def test_empty_features_sets_single_zero_feature(self):
        g = _citation_like_graph()
        out = DGLEmptyFeatures()(g)
        assert out.num_edges() == g.num_edges()
        assert out.ndata["x"].shape == (g.num_nodes(), 1)
        assert torch.all(out.ndata["x"] == 0)

    def test_random_features_shuffle_preserves_feature_multiset(self):
        g = _citation_like_graph()
        out = DGLRandomFeatures(shuffle=True)(g)
        assert out.ndata["x"].shape == g.ndata["x"].shape

        original_rows = Counter(tuple(row.tolist()) for row in g.ndata["x"])
        shuffled_rows = Counter(tuple(row.tolist()) for row in out.ndata["x"])
        assert original_rows == shuffled_rows

    def test_random_features_respects_seeded_global_rng(self):
        g = _citation_like_graph()

        torch.manual_seed(123)
        out1 = DGLRandomFeatures(shuffle=False)(g).ndata["x"]
        torch.manual_seed(123)
        out2 = DGLRandomFeatures(shuffle=False)(g).ndata["x"]

        assert torch.allclose(out1, out2)

    def test_random_features_shuffle_with_missing_feature_is_noop(self):
        g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 0])), num_nodes=2)
        out = DGLRandomFeatures(shuffle=True, feat_name="missing")(g)
        assert out.num_nodes() == 2
        assert "missing" not in out.ndata

    def test_empty_graph_removes_edges_but_keeps_node_features(self):
        g = _citation_like_graph()
        out = DGLEmptyGraph()(g)
        assert out.num_nodes() == g.num_nodes()
        assert out.num_edges() == 0
        assert torch.equal(out.ndata["x"], g.ndata["x"])

    def test_complete_graph_has_all_directed_edges_without_self_loops(self):
        g = _citation_like_graph()
        out = DGLCompleteGraph()(g)
        n = g.num_nodes()

        assert out.num_edges() == n * (n - 1)
        src, dst = out.edges()
        assert torch.all(src != dst)
        assert torch.equal(out.ndata["x"], g.ndata["x"])

    def test_random_graph_has_no_self_loops_and_keeps_features(self):
        g = _citation_like_graph()
        out = DGLRandomGraph(p=0.4)(g)
        src, dst = out.edges()
        assert out.num_nodes() == g.num_nodes()
        assert torch.all(src != dst)
        assert torch.equal(out.ndata["x"], g.ndata["x"])

    def test_study_apply_supports_dgl_graph(self):
        g = _citation_like_graph()
        out = SeparabilityStudy.apply(g, DGLEmptyGraph())
        assert out.num_nodes() == g.num_nodes()
        assert out.num_edges() == 0

    def test_adapter_preserves_non_feature_ndata(self):
        g = _citation_like_graph()
        g.ndata["y"] = torch.arange(g.num_nodes(), dtype=torch.float32).unsqueeze(1)
        out = DGLEmptyGraph()(g)
        assert "y" in out.ndata
        assert torch.equal(out.ndata["y"], g.ndata["y"])

    def test_as_dgl_transform_supports_custom_feature_key(self):
        g = dgl.graph(
            (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])),
            num_nodes=3,
        )
        g.ndata["h"] = torch.tensor(
            [[3.0, 1.0], [1.0, 4.0], [2.0, 2.0]], dtype=torch.float32
        )
        wrapped = as_dgl_transform(EmptyFeatures(), feat_name="h")
        out = wrapped(g)
        assert "h" in out.ndata
        assert out.ndata["h"].shape == (3, 1)
        assert torch.all(out.ndata["h"] == 0)
