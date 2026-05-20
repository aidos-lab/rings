import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from rings import EmptyGraph, Original
from rings.integrations import SeparabilityStudy
from rings.separability.comparator import KSComparator, WilcoxonComparator


def _toy_data():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return Data(x=x, edge_index=edge_index)


class TestSeparabilityStudy:
    def test_init_empty_perturbations_raises(self):
        with pytest.raises(ValueError):
            SeparabilityStudy(perturbations={})

    def test_comparator_string_shortcuts(self):
        s = SeparabilityStudy(
            perturbations={"Original": Original()}, comparator="ks"
        )
        assert isinstance(s.comparator, KSComparator)
        s = SeparabilityStudy(
            perturbations={"Original": Original()}, comparator="wilcoxon"
        )
        assert isinstance(s.comparator, WilcoxonComparator)

    def test_comparator_unknown_string_raises(self):
        with pytest.raises(ValueError):
            SeparabilityStudy(
                perturbations={"Original": Original()}, comparator="bogus"
            )

    def test_runs_cartesian_product(self):
        study = SeparabilityStudy(
            perturbations={"A": Original(), "B": EmptyGraph()}, num_seeds=3
        )
        triples = list(study.runs())
        assert len(triples) == 6
        names = [t[0] for t in triples]
        assert names.count("A") == 3 and names.count("B") == 3
        seeds = sorted({t[2] for t in triples})
        assert seeds == [0, 1, 2]

    def test_apply_on_single_data_returns_transformed(self):
        data = _toy_data()
        out = SeparabilityStudy.apply(data, EmptyGraph())
        # EmptyGraph should strip edges
        assert out.edge_index.shape[1] == 0

    def test_apply_on_dataset_sets_transform_attribute(self):
        # Use a real PyG Dataset (TUDataset cached avoidance: build minimal fake via subclass).
        from torch_geometric.data import InMemoryDataset

        class TinyDataset(InMemoryDataset):
            def __init__(self):
                super().__init__(root=None)
                self._data_list = [_toy_data(), _toy_data()]

            def len(self):
                return len(self._data_list)

            def get(self, idx):
                return self._data_list[idx]

        ds = TinyDataset()
        transform = EmptyGraph()
        out = SeparabilityStudy.apply(ds, transform)
        assert out is ds
        assert ds.transform is transform

    def test_record_unknown_name_raises(self):
        study = SeparabilityStudy(perturbations={"A": Original()})
        with pytest.raises(KeyError):
            study.record("B", 0.5)

    def test_evaluate_without_scores_raises(self):
        study = SeparabilityStudy(perturbations={"A": Original()})
        with pytest.raises(RuntimeError):
            study.evaluate()

    def test_record_and_evaluate_produces_dataframe(self):
        study = SeparabilityStudy(
            perturbations={"A": Original(), "B": EmptyGraph()},
            num_seeds=10,
            alpha=0.05,
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            study.record("A", float(rng.normal(0.9, 0.02)))
            study.record("B", float(rng.normal(0.5, 0.02)))

        results = study.evaluate(n_permutations=200)
        assert len(results) == 1
        for col in ("mode1", "mode2", "significant"):
            assert col in results.columns
        assert bool(results.iloc[0]["significant"]) is True
