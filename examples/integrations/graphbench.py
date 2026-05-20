"""
RINGS separability for a GraphBench-evaluated PyG training pipeline.

This example shows the canonical recipe when you want GraphBench metrics on a
standard graph-classification benchmark (MUTAG) while keeping the same GCN training
loop as the other integration examples:

1. Load MUTAG with PyG's ``TUDataset`` (GraphBench does not ship TU benchmarks).
2. Loop over perturbations x seeds via ``SeparabilityStudy.runs()``.
3. Apply each perturbation to the base dataset and train a GCN with an ordinary
   PyG loop (same architecture and hyperparameters as ``lightning.py``).
4. Score the test set with ``graphbench.Evaluator`` (``algoreas_classification`` → ACC).
5. Call ``study.evaluate()`` for the pairwise separability table.

Usage:
    uv run --with graphbench-lib python -m examples.integrations.graphbench
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import graphbench

from rings import CompleteFeatures, EmptyGraph, Original, RandomFeatures
from rings.integrations import SeparabilityStudy


class SimpleGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


def train_and_eval(dataset, seed, evaluator, epochs=20):
    """Train a GCN on MUTAG and return GraphBench ACC on the held-out split."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(dataset)
    split = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_ds = dataset[perm[:split]]
    test_ds = dataset[perm[split:]]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = SimpleGCN(
        hidden_channels=64,
        num_node_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data in train_loader:
            data = data.to(device)
            optim.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss_fn(out, data.y).backward()
            optim.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            preds.append(pred.cpu())
            labels.append(data.y.cpu())

    y_pred = torch.cat(preds).unsqueeze(1)
    y_true = torch.cat(labels).unsqueeze(1)
    metrics = evaluator.evaluate(y_pred, y_true)
    # algoreas_classification reports ACC and F1; keep ACC to mirror lightning's test_acc.
    return metrics[0] if isinstance(metrics, list) else metrics


def main():
    base_dataset = TUDataset(root="data/TUDataset", name="MUTAG")
    max_nodes = max(g.num_nodes for g in base_dataset)
    evaluator = graphbench.Evaluator("algoreas_classification")

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
        perturbed = study.apply(base_dataset, transform)
        score = train_and_eval(perturbed, seed=seed, evaluator=evaluator)
        study.record(name, score)
        print(f"  [{name} seed={seed}] test_acc={score:.3f}")

    print("\nSeparability:")
    print(
        study.evaluate(n_permutations=1000)[
            ["mode1", "mode2", "score", "pvalue_adjusted", "significant"]
        ]
    )


if __name__ == "__main__":
    main()
