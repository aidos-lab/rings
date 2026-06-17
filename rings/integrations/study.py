"""Lightweight collector for running RINGS separability studies inside an existing pipeline.

``SeparabilityStudy`` is intentionally framework-agnostic: it does not own the training
loop, the dataset loader, or the evaluator. The user drives those — the study just
holds the perturbation set, hands out ``(name, transform, seed)`` triples to iterate
over, applies a transform to a PyG ``Data`` or ``Dataset``, records scalar scores,
and runs ``SeparabilityFunctor`` over the collected distributions.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from rings.separability.comparator import KSComparator, WilcoxonComparator
from rings.separability.functor import SeparabilityFunctor


class SeparabilityStudy:
    """
    Collect per-perturbation, per-seed scores from a user-driven training loop and
    compute pairwise separability across perturbations.

    Parameters
    ----------
    perturbations : Dict[str, Callable]
        Mapping of perturbation name to a PyG ``BaseTransform`` (e.g. ``Original()``,
        ``EmptyGraph()``).
    num_seeds : int, default=5
        Number of seeds to iterate per perturbation. The seed values yielded are
        ``range(num_seeds)``; the user is responsible for using them to seed any
        framework RNGs inside their loop.
    comparator : str or Callable, default="ks"
        Either ``"ks"`` / ``"wilcoxon"`` or a comparator instance passed straight
        to ``SeparabilityFunctor``.
    alpha : float, default=0.01
        Family-wise significance level for the separability test.
    n_jobs : int, default=1
        Forwarded to ``SeparabilityFunctor`` for parallel pairwise comparison.

    Examples
    --------
    >>> from rings import Original, EmptyGraph
    >>> from rings.integrations import SeparabilityStudy
    >>> study = SeparabilityStudy(
    ...     perturbations={"Original": Original(), "EmptyGraph": EmptyGraph()},
    ...     num_seeds=5,
    ... )
    >>> for name, transform, seed in study.runs():
    ...     dataset = study.apply(base_dataset, transform)
    ...     score = my_train_and_eval(dataset, seed=seed)
    ...     study.record(name, score)
    >>> results = study.evaluate()
    """

    def __init__(
        self,
        perturbations: Dict[str, Callable],
        num_seeds: int = 5,
        comparator: Union[str, Callable] = "ks",
        alpha: float = 0.01,
        n_jobs: int = 1,
    ):
        if not perturbations:
            raise ValueError("`perturbations` must contain at least one entry.")

        self.perturbations = perturbations
        self.num_seeds = num_seeds
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.comparator = self._resolve_comparator(comparator)
        self._scores: Dict[str, List[float]] = defaultdict(list)

    @staticmethod
    def _resolve_comparator(comparator: Union[str, Callable]) -> Callable:
        if not isinstance(comparator, str):
            return comparator
        key = comparator.lower()
        if key == "ks":
            return KSComparator()
        if key == "wilcoxon":
            return WilcoxonComparator()
        raise ValueError(
            f"Unknown comparator '{comparator}'. Use 'ks', 'wilcoxon', or a comparator instance."
        )

    def runs(self) -> Iterator[Tuple[str, Callable, int]]:
        """Yield ``(perturbation_name, transform, seed)`` for every perturbation × seed."""
        for name, transform in self.perturbations.items():
            for seed in range(self.num_seeds):
                yield name, transform, seed

    @staticmethod
    def apply(data: Any, transform: Callable) -> Any:
        """Apply ``transform`` to a PyG ``Data``/``Dataset`` or a DGL ``DGLGraph``.

        For a PyG ``Dataset``, this sets ``dataset.transform`` for lazy application.
        For PyG ``Data`` or DGL ``DGLGraph``, the transform is called directly.
        """
        # Try PyG
        try:
            from torch_geometric.data import Data, Dataset

            if isinstance(data, Dataset):
                data.transform = transform
                return data
            if isinstance(data, Data):
                return transform(data)
        except ImportError:
            pass

        # Try DGL
        try:
            import dgl

            if isinstance(data, dgl.DGLGraph):
                return transform(data)
        except ImportError:
            pass

        # Fallback for generic objects
        return transform(data)

    def record(self, name: str, score: float) -> None:
        """Record a scalar score for ``name``."""
        if name not in self.perturbations:
            raise KeyError(
                f"'{name}' is not a registered perturbation. Known: {list(self.perturbations)}"
            )
        self._scores[name].append(float(score))

    @property
    def scores(self) -> Dict[str, np.ndarray]:
        """Recorded scores keyed by perturbation name."""
        return {name: np.asarray(vals) for name, vals in self._scores.items()}

    def evaluate(
        self,
        n_permutations: int = 10_000,
        random_state: Optional[int] = 42,
        as_dataframe: bool = True,
    ):
        """Run pairwise separability on the recorded distributions."""
        if not self._scores:
            raise RuntimeError(
                "No scores recorded. Call `record(name, score)` inside your training loop "
                "before calling `evaluate()`."
            )
        functor = SeparabilityFunctor(
            comparator=self.comparator,
            n_jobs=self.n_jobs,
            alpha=self.alpha,
        )
        return functor.forward(
            distributions=self.scores,
            n_permutations=n_permutations,
            random_state=random_state,
            as_dataframe=as_dataframe,
        )
