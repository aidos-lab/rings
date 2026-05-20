🔑 Performance Separability
============================

The ``rings.separability`` module tests whether performance distributions from different perturbations are statistically distinguishable. It pairs a **comparator** (the statistical test) with a **functor** (the orchestration: pairwise comparison, permutation testing, Bonferroni correction).

|

.. image:: _static/separability-overview.svg
   :width: 600
   :alt: Performance separability overview
   :align: center

|

Comparators
-----------

.. image:: _static/separability-comparator.svg
   :width: 500
   :alt: Comparator
   :align: center

|

.. automodule:: rings.separability.comparator
   :members:

Functor
-------

.. image:: _static/separability-functor.svg
   :width: 500
   :alt: Functor
   :align: center

|

.. automodule:: rings.separability.functor
   :members:
