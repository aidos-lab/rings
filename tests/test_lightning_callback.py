import pytest

pytest.importorskip("pytorch_lightning")

import torch

from rings import EmptyGraph, Original
from rings.integrations import SeparabilityCallback, SeparabilityStudy


class _StubTrainer:
    def __init__(self, metrics):
        self.callback_metrics = metrics


class TestSeparabilityCallback:
    def test_records_metric_into_study(self):
        study = SeparabilityStudy(perturbations={"Original": Original()})
        cb = SeparabilityCallback(study, "Original", metric_key="test_acc")
        trainer = _StubTrainer({"test_acc": torch.tensor(0.87)})

        cb.on_test_end(trainer, pl_module=None)

        assert study.scores["Original"].tolist() == [pytest.approx(0.87, rel=1e-5)]

    def test_unknown_perturbation_raises(self):
        study = SeparabilityStudy(perturbations={"Original": Original()})
        with pytest.raises(KeyError):
            SeparabilityCallback(study, "NotRegistered")

    def test_missing_metric_raises(self):
        study = SeparabilityStudy(perturbations={"Original": Original()})
        cb = SeparabilityCallback(study, "Original", metric_key="test_acc")
        trainer = _StubTrainer({"other_metric": torch.tensor(0.5)})

        with pytest.raises(KeyError, match="test_acc"):
            cb.on_test_end(trainer, pl_module=None)

    def test_accumulates_across_calls(self):
        study = SeparabilityStudy(
            perturbations={"A": Original(), "B": EmptyGraph()}, num_seeds=2
        )
        cb_a = SeparabilityCallback(study, "A")
        cb_b = SeparabilityCallback(study, "B")

        cb_a.on_test_end(_StubTrainer({"test_acc": torch.tensor(0.9)}), None)
        cb_a.on_test_end(_StubTrainer({"test_acc": torch.tensor(0.85)}), None)
        cb_b.on_test_end(_StubTrainer({"test_acc": torch.tensor(0.5)}), None)
        cb_b.on_test_end(_StubTrainer({"test_acc": torch.tensor(0.55)}), None)

        assert study.scores["A"].tolist() == [
            pytest.approx(0.9, rel=1e-5),
            pytest.approx(0.85, rel=1e-5),
        ]
        assert study.scores["B"].tolist() == [
            pytest.approx(0.5, rel=1e-5),
            pytest.approx(0.55, rel=1e-5),
        ]

    def test_accepts_plain_float_metric(self):
        study = SeparabilityStudy(perturbations={"Original": Original()})
        cb = SeparabilityCallback(study, "Original")
        trainer = _StubTrainer({"test_acc": 0.42})

        cb.on_test_end(trainer, pl_module=None)
        assert study.scores["Original"].tolist() == [pytest.approx(0.42, rel=1e-5)]
