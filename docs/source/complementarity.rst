🔑 Mode Complementarity
========================

The ``rings.complementarity`` module measures the alignment between node features and graph structure by comparing their induced metric spaces. It pairs **graph metrics** (diffusion, heat-kernel, resistance, shortest-path) with **matrix-norm comparators** and a **functor** that handles disconnected graphs component-wise.

|

.. image:: _static/complementarity-overview.svg
   :width: 600
   :alt: Mode complementarity overview
   :align: center

|

Functor
-------

.. image:: _static/complementarity-functor.svg
   :width: 500
   :alt: Functor
   :align: center

|

.. automodule:: rings.complementarity.functor
   :members:

Comparators
-----------

.. image:: _static/complementarity-comparator.svg
   :width: 500
   :alt: Comparator
   :align: center

|

.. automodule:: rings.complementarity.comparator
   :members:

Metrics
-------

.. image:: _static/complementarity-metrics.svg
   :width: 500
   :alt: Metrics
   :align: center

|

.. automodule:: rings.complementarity.metrics
   :members:

Utilities
---------

.. automodule:: rings.complementarity.utils
   :members:
