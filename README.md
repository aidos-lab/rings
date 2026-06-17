# RINGS

**[No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets](https://arxiv.org/abs/2502.02379)** — ICML 2025.

RINGS is a perturbation framework for attributed graphs that lets you evaluate graph-learning datasets and models from first principles: apply structured perturbations, train as usual, and compare performance distributions with statistically rigorous tests.

---

## Install

Install from [PyPI](https://pypi.org/project/rings-evaluation/):

```bash
pip install rings-evaluation
```

Requires Python 3.11+.

### Optional integrations

Install only what you need:

```bash
# PyTorch Lightning integration
pip install "rings-evaluation[lightning]"

# DGL integration (available for Python < 3.13)
pip install "rings-evaluation[dgl]"

# Both integrations
pip install "rings-evaluation[integrations]"
```

### From source

To contribute or run the examples in this repo:

```bash
pip install uv
git clone https://github.com/aidos-lab/rings.git && cd rings
uv sync && source .venv/bin/activate
```

Enable optional integration groups as needed:

```bash
uv sync --group lightning
uv sync --group dgl
uv sync --group integrations
```

---

## Quickstart 

> Drop RINGS evaluations into your GNN pipeline.

Keep your training loop. Wrap it with `SeparabilityStudy` to iterate perturbation × seed, record one scalar per run from *your* evaluator, and get a pairwise separability table back.

```python
from rings import Original, EmptyGraph, RandomFeatures, CompleteFeatures
from rings.integrations import SeparabilityStudy

study = SeparabilityStudy(
    perturbations={
        "Original":         Original(),
        "EmptyGraph":       EmptyGraph(),
        "RandomFeatures":   RandomFeatures(shuffle=True),
        "CompleteFeatures": CompleteFeatures(max_nodes=max_nodes),
    },
    num_seeds=5,
    comparator="ks",   # or "wilcoxon"
    alpha=0.05,
)

for name, transform, seed in study.runs():
    perturbed = study.apply(base_dataset, transform)
    score = train_and_eval(perturbed, seed=seed)   # your code
    study.record(name, score)

results = study.evaluate(n_permutations=1000)
# DataFrame: mode1, mode2, score, pvalue_adjusted, significant
```

**PyTorch Lightning** — attach `SeparabilityCallback` to your `Trainer` and it records the logged `test_acc` automatically:

```python
import pytorch_lightning as pl
from rings.integrations import SeparabilityStudy, SeparabilityCallback

for name, transform, seed in study.runs():
    pl.seed_everything(seed, workers=True)
    dm = make_datamodule(study.apply(base_dataset, transform), seed=seed)
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[SeparabilityCallback(study, perturbation_name=name)],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

results = study.evaluate()
```

**DGL** — use the DGL wrappers from `rings.integrations` (backed by the same core perturbation logic):

```python
from rings.integrations import DGLOriginal, DGLEmptyGraph, SeparabilityStudy

study = SeparabilityStudy(
    perturbations={
        "Original": DGLOriginal(),
        "EmptyGraph": DGLEmptyGraph(),
    },
    num_seeds=5,
)

for name, transform, seed in study.runs():
    perturbed = study.apply(base_dgl_graph, transform)
    score = train_and_eval_dgl(perturbed, seed=seed)  # your code
    study.record(name, score)

results = study.evaluate()
```

**Custom evaluator** (GraphBench, OGB, anything that returns a scalar): just pass the number to `study.record(name, score)`.

### Runnable examples

```bash
uv run -m examples.integrations.pyg
uv run --with lightning -m examples.integrations.lightning
uv run --with graphbench-lib -m examples.integrations.graphbench
```

---

## Learn more

- **[Perturbations](https://aidos-lab.github.io/rings/perturbations.html)** — node-feature and graph-structure transforms (`Original`, `EmptyGraph`, `RandomFeatures`, `CompleteFeatures`, `RandomConnectedGraph`, …)
- **[SeparabilityFunctor](https://aidos-lab.github.io/rings/separability/functor.html)** — pairwise distribution tests (KS, Wilcoxon) with permutation p-values and Bonferroni correction
- **[ComplementarityFunctor](https://aidos-lab.github.io/rings/complementarity/functor.html)** — metric-space alignment between node features and graph structure
- **[Integrations API](https://aidos-lab.github.io/rings/integrations.html)** — full reference for `SeparabilityStudy` and `SeparabilityCallback`
- **`examples/`** — end-to-end scripts for separability, complementarity, and the three integration recipes

---

## Citation

```bibtex
@inproceedings{coupette2025metric,
  title     = {No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets},
  author    = {Corinna Coupette and Jeremy Wayland and Emily Simons and Bastian Rieck},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=XbmBNwrfG5}
}
```
