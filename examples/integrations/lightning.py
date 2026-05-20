"""
RINGS separability for a PyTorch Lightning training pipeline.

This example shows the canonical recipe when you're using Lightning:

1. Wrap your existing GCN as a ``LightningModule``.
2. Loop over perturbations x seeds via ``SeparabilityStudy.runs()``.
3. Apply each perturbation to your base dataset, wrap it in
   ``torch_geometric.data.lightning.LightningDataset``, and run
   ``Trainer.fit(...) / Trainer.test(...)`` exactly as you normally would.
4. Attach ``SeparabilityCallback`` to the test ``Trainer`` so the logged
   ``test_acc`` is recorded into the study automatically.
5. Call ``study.evaluate()`` for the pairwise separability table.

Usage:
    pip install -e .[lightning]
    uv run python examples/lightning_separability.py
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool

from rings import CompleteFeatures, EmptyGraph, Original, RandomFeatures
from rings.integrations import SeparabilityCallback, SeparabilityStudy


class GCNClassifier(pl.LightningModule):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.lr = lr

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

    def training_step(self, batch, _):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch, _):
        pred = self(batch).argmax(dim=1)
        acc = (pred == batch.y).float().mean()
        self.log("test_acc", acc, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def make_datamodule(dataset, seed: int, batch_size: int = 32) -> LightningDataset:
    n = len(dataset)
    split = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_ds = dataset[perm[:split]]
    test_ds = dataset[perm[split:]]
    return LightningDataset(
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=batch_size,
    )


def main():
    base_dataset = TUDataset(root="data/TUDataset", name="MUTAG")
    max_nodes = max(g.num_nodes for g in base_dataset)

    study = SeparabilityStudy(
        perturbations={
            "Original": Original(),
            "EmptyGraph": EmptyGraph(),
            "RandomFeatures": RandomFeatures(shuffle=True),
            "CompleteFeatures": CompleteFeatures(max_nodes=max_nodes),
        },
        num_seeds=5,
        comparator="ks",
        alpha=0.05,
    )

    for name, transform, seed in study.runs():
        pl.seed_everything(seed, workers=True)
        perturbed = study.apply(base_dataset, transform)
        dm = make_datamodule(perturbed, seed=seed)

        model = GCNClassifier(
            num_node_features=base_dataset.num_node_features,
            num_classes=base_dataset.num_classes,
        )
        trainer = pl.Trainer(
            max_epochs=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[SeparabilityCallback(study, perturbation_name=name)],
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, verbose=False)
        print(f"  [{name} seed={seed}] test_acc={study.scores[name][-1]:.3f}")

    print("\nSeparability:")
    print(
        study.evaluate(n_permutations=1000)[
            ["mode1", "mode2", "score", "pvalue_adjusted", "significant"]
        ]
    )


if __name__ == "__main__":
    main()
