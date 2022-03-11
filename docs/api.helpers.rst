.. _api_helpers:

.. currentmodule:: nanomesh

Helper classes and functions
============================

These are helper functions for working with meshes. A :class:`RegionMarker`
stores metadata about a region in a mesh. :func:`tetrahedralize` and
:func:`triangulate` are interfaces to generate meshes.

.. rubric:: classes

.. autosummary::

   nanomesh.RegionMarker

.. rubric:: functions

.. autosummary::

   nanomesh.tetrahedralize
   nanomesh.simple_triangulate
   nanomesh.triangulate

Reference
---------

.. autoclass:: RegionMarker

.. autofunction:: tetrahedralize
.. autofunction:: triangulate
