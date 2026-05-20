"""PyTorch Lightning callback for recording test-time metrics into a SeparabilityStudy."""

from typing import TYPE_CHECKING

from rings.integrations.study import SeparabilityStudy

try:
    import pytorch_lightning as _pl
except ImportError:
    try:
        import lightning.pytorch as _pl
    except ImportError:
        _pl = None

if TYPE_CHECKING:
    import pytorch_lightning as _pl_typing  # noqa: F401

_HAS_LIGHTNING = _pl is not None
_Callback = _pl.Callback if _HAS_LIGHTNING else object


class SeparabilityCallback(_Callback):
    """
    Record a Lightning test metric into a :class:`SeparabilityStudy` once per ``trainer.test()`` call.

    Attach one of these per perturbation run. On ``on_test_end`` it reads
    ``trainer.callback_metrics[metric_key]`` and appends the scalar value to the
    study under ``perturbation_name``. After looping over all perturbation x seed
    combinations, call ``study.evaluate()`` to get the separability DataFrame.

    Parameters
    ----------
    study : SeparabilityStudy
        The study to record into.
    perturbation_name : str
        Which perturbation this run corresponds to. Must already be a key in
        ``study.perturbations``.
    metric_key : str, default="test_acc"
        Key under which the test metric is logged in ``trainer.callback_metrics``.
        The user's ``LightningModule.test_step`` (or ``test_epoch_end``) must call
        ``self.log(metric_key, value)`` for this to work.

    Examples
    --------
    >>> from rings.integrations import SeparabilityStudy, SeparabilityCallback
    >>> study = SeparabilityStudy(perturbations={"Original": Original(), ...})
    >>> for name, transform, seed in study.runs():
    ...     dm = build_data_module(study.apply(base_dataset, transform), seed)
    ...     trainer = pl.Trainer(callbacks=[SeparabilityCallback(study, name)])
    ...     trainer.fit(model, dm)
    ...     trainer.test(model, dm)
    >>> results = study.evaluate()
    """

    def __init__(
        self,
        study: SeparabilityStudy,
        perturbation_name: str,
        metric_key: str = "test_acc",
    ):
        if not _HAS_LIGHTNING:
            raise ImportError(
                "pytorch-lightning is required for SeparabilityCallback. "
                "Install it with 'pip install rings[lightning]' or 'pip install pytorch-lightning'."
            )
        if perturbation_name not in study.perturbations:
            raise KeyError(
                f"'{perturbation_name}' is not a registered perturbation in the study. "
                f"Known: {list(study.perturbations)}"
            )
        super().__init__()
        self.study = study
        self.perturbation_name = perturbation_name
        self.metric_key = metric_key

    def on_test_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        if self.metric_key not in metrics:
            raise KeyError(
                f"Metric '{self.metric_key}' was not logged during testing. "
                f"Available keys: {list(metrics)}. "
                f"Ensure your LightningModule calls `self.log('{self.metric_key}', ...)`."
            )
        value = metrics[self.metric_key]
        score = float(value.item() if hasattr(value, "item") else value)
        self.study.record(self.perturbation_name, score)
