import pytest

pl = pytest.importorskip("pytorch_lightning")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torch_geometric.datasets import KarateClub  # noqa: E402

from rings import EmptyGraph, Original  # noqa: E402
from rings.integrations import (  # noqa: E402
    DGLEmptyGraph,
    DGLOriginal,
    SeparabilityCallback,
    SeparabilityStudy,
)


class _SingleGraphDataset(Dataset):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph


def _single_graph_loader(graph):
    dataset = _SingleGraphDataset(graph)
    return DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])


class _GraphScoreModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def configure_optimizers(self):
        return torch.optim.SGD([self._dummy], lr=0.1)

    def test_step(self, batch, batch_idx):
        if hasattr(batch, "edge_index"):
            num_nodes = int(batch.num_nodes)
            num_edges = int(batch.edge_index.size(1))
        else:
            num_nodes = int(batch.num_nodes())
            num_edges = int(batch.num_edges())
        score = float(num_edges) / max(float(num_nodes), 1.0)
        self.log("test_acc", score, prog_bar=False, on_step=False, on_epoch=True)
        return score


def _run_study_with_lightning(base_graph, perturbations, num_seeds=2):
    study = SeparabilityStudy(perturbations=perturbations, num_seeds=num_seeds)
    model = _GraphScoreModule()

    for name, transform, seed in study.runs():
        torch.manual_seed(seed)
        graph = study.apply(base_graph, transform)
        callback = SeparabilityCallback(study, perturbation_name=name, metric_key="test_acc")
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[callback],
        )
        trainer.test(model, dataloaders=_single_graph_loader(graph), verbose=False)
    return study


def _load_dgl_karateclub_or_skip():
    try:
        import dgl  # noqa: F401
        from dgl.data import KarateClubDataset
    except Exception as exc:
        pytest.skip(f"DGL unavailable in test runtime: {exc}")
    return KarateClubDataset()[0]


def test_lightning_callback_with_pyg_karateclub():
    base_graph = KarateClub()[0]
    study = _run_study_with_lightning(
        base_graph=base_graph,
        perturbations={"Original": Original(), "EmptyGraph": EmptyGraph()},
        num_seeds=2,
    )

    assert len(study.scores["Original"]) == 2
    assert len(study.scores["EmptyGraph"]) == 2
    assert all(score > 0 for score in study.scores["Original"])
    assert all(score == 0 for score in study.scores["EmptyGraph"])


def test_lightning_callback_with_dgl_karateclub():
    base_graph = _load_dgl_karateclub_or_skip()
    if "x" not in base_graph.ndata:
        if "feat" not in base_graph.ndata:
            pytest.skip("KarateClubDataset does not expose expected node features.")
        base_graph.ndata["x"] = base_graph.ndata["feat"].float()

    study = _run_study_with_lightning(
        base_graph=base_graph,
        perturbations={"Original": DGLOriginal(), "EmptyGraph": DGLEmptyGraph()},
        num_seeds=2,
    )

    assert len(study.scores["Original"]) == 2
    assert len(study.scores["EmptyGraph"]) == 2
    assert all(score > 0 for score in study.scores["Original"])
    assert all(score == 0 for score in study.scores["EmptyGraph"])
