.. _api_meshing:

.. currentmodule:: nanomesh

Meshing routines
================

:func:`plane2mesh` and :func:`volume2mesh` are high-level functions for
generating a mesh. For finer control, use the classes :class:`Mesher2D` for
image data and :class:`Mesher3D` for volume data.

Both classes derive from :class:`Mesher`. Initiating a :class:`Mesher` instance
will create the appropriate meshing class.

.. rubric:: Classes

.. autosummary::

   nanomesh.Mesher
   nanomesh.Mesher2D
   nanomesh.Mesher3D

.. rubric:: Functions

.. autosummary::

   nanomesh.plane2mesh
   nanomesh.volume2mesh

Reference
---------

.. autofunction:: plane2mesh
.. autofunction:: volume2mesh

.. autoclass:: Mesher
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: Mesher2D
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: Mesher3D
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
