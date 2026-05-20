🔌 Integrations
================

RINGS can be easily integrated into your GNN training pipeline.

The ``rings.integrations`` module ships two small utilities:

- :class:`~rings.integrations.study.SeparabilityStudy` — a collector that iterates perturbation × seed, applies transforms PyG-idiomatically, records scalar scores from *your* evaluator, and returns a pairwise separability DataFrame. Use it with plain PyG, Lightning, or any other framework.
- :class:`~rings.integrations.lightning.SeparabilityCallback` — a PyTorch Lightning specific callback that records a logged test metric into a study automatically at the end of ``trainer.test()``.

Plain PyG
---------

.. code-block:: python

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

Lightning
---------

.. code-block:: python

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

Your ``LightningModule.test_step`` must call ``self.log("test_acc", acc)`` (or whatever ``metric_key`` you pass to ``SeparabilityCallback``).

Custom evaluators
-----------------

``study.record(name, score)`` accepts any scalar — plug in `GraphBench <https://github.com/aidos-lab/graphbench>`_, OGB evaluators, or your own metric. See ``examples/integrations/graphbench.py``.

Runnable recipes
----------------

.. code-block:: bash

   uv run -m examples.integrations.pyg
   uv run --with lightning -m examples.integrations.lightning
   uv run --with graphbench-lib -m examples.integrations.graphbench

API reference
-------------

SeparabilityStudy
^^^^^^^^^^^^^^^^^

.. automodule:: rings.integrations.study
   :members:
   :undoc-members:
   :show-inheritance:

SeparabilityCallback
^^^^^^^^^^^^^^^^^^^^

.. automodule:: rings.integrations.lightning
   :members:
   :undoc-members:
   :show-inheritance:
