.. _api_metrics:

.. module:: nanomesh.metrics

Metrics
=======

The :mod:`nanomesh.metrics` module helps with calculating different types of cell metrics.

There are a few higher level functions available. While one could use
:func:`calculate_all_metrics` to calculate all available metrics,
each function is also available by itself.

:func:`histogram` and :func:`plot2d` are helpers
to visualize the metrics.

.. seealso::

    For more info, see the example on :doc:`examples/calculate_cell_metrics`.

.. rubric:: Functions

These metrics are currently available:

.. autosummary::

   area
   aspect_frobenius
   aspect_ratio
   condition
   distortion
   max_angle
   max_min_edge_ratio
   min_angle
   radius_ratio
   relative_size_squared
   scaled_jacobian
   shape
   shape_and_size

Other functions:

.. autosummary::

   calculate_all_metrics
   histogram
   plot2d

Reference
---------

.. autofunction:: calculate_all_metrics
.. autofunction:: histogram
.. autofunction:: plot2d

.. autofunction:: area
.. autofunction:: aspect_frobenius
.. autofunction:: aspect_ratio
.. autofunction:: condition
.. autofunction:: distortion
.. autofunction:: max_angle
.. autofunction:: max_min_edge_ratio
.. autofunction:: min_angle
.. autofunction:: radius_ratio
.. autofunction:: relative_size_squared
.. autofunction:: scaled_jacobian
.. autofunction:: shape
.. autofunction:: shape_and_size
