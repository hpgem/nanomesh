.. _api:

Python Interface
================

.. module:: nanomesh

This part of the documentation covers the public interface of Nanomesh.

Most of Nanomesh' functionality can be accessed through the classes
and functions listed below. See the :ref:`examples` for how to use them.

Working with image data
-----------------------

Nanomesh has two classes for representing image data, :class:`Plane` for working
with 2D pixel data and :class:`Volume` for working with 3D voxel data.
These can be used to load, crop, transform, filter, and segment image data.

.. autoclass:: Plane
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: Volume
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Working with mesh data
----------------------

A large part of Nanomesh deals with with generating and manipulating
2D and 3D meshes. To store the data, Nanomesh uses :class:`MeshContainer`,
which is a somewhat low-level, generic container for mesh data. It can
store different types of cells and associated data. It is used
to read/write data via `meshio <https://github.com/nschloe/meshio>`_.

To deal with mesh data more directly, use :class:`LineMesh`,
:class:`TriangleMesh` or :class:`TetraMesh`. These contain
dedicated methods for working with a specific type of mesh and plotting them.
Each can be generated from :class:`MeshContainer`.

.. autoclass:: MeshContainer
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: LineMesh
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: TriangleMesh
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: TetraMesh
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Making a mesh out of image data
-------------------------------

:func:`plane2mesh` and :func:`volume2mesh` are high-level functions for
generating a mesh. For finer control, use the classes :class:`Mesher2D` for
image data and :class:`Mesher3D` for volume data.

.. autofunction:: plane2mesh
.. autofunction:: volume2mesh

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

Helper classes and functions
----------------------------

These are helper functions for working with meshes. A :class:`RegionMarker`
stores metadata about a region in a mesh. :func:`tetrahedralize` and
:func:`triangulate` are interfaces to generate meshes.

.. autoclass:: RegionMarker

.. autofunction:: tetrahedralize
.. autofunction:: triangulate


Metrics
-------

The :mod:`nanomesh.metrics` module helps with calculating different types of cell metrics.

There are a few higher level functions available. While one could use
:func:`~nanomesh.metrics.calculate_all_metrics` to calculate all available metrics,
each function is also available by itself.

These metrics are currently available:

- :func:`~nanomesh.metrics.area`
- :func:`~nanomesh.metrics.aspect_frobenius`
- :func:`~nanomesh.metrics.aspect_ratio`
- :func:`~nanomesh.metrics.condition`
- :func:`~nanomesh.metrics.distortion`
- :func:`~nanomesh.metrics.max_angle`
- :func:`~nanomesh.metrics.max_min_edge_ratio`
- :func:`~nanomesh.metrics.min_angle`
- :func:`~nanomesh.metrics.radius_ratio`
- :func:`~nanomesh.metrics.relative_size_squared`
- :func:`~nanomesh.metrics.scaled_jacobian`
- :func:`~nanomesh.metrics.shape`
- :func:`~nanomesh.metrics.shape_and_size`

:func:`~nanomesh.metrics.histogram` and :func:`~nanomesh.metrics.plot2d` are helpers
to visualize the metrics.

For more info, see <<TODO: LINK TO EXAMPLE WITH METRICS>>.

.. autofunction:: nanomesh.metrics.calculate_all_metrics
.. autofunction:: nanomesh.metrics.histogram
.. autofunction:: nanomesh.metrics.plot2d

.. autofunction:: nanomesh.metrics.area
.. autofunction:: nanomesh.metrics.aspect_frobenius
.. autofunction:: nanomesh.metrics.aspect_ratio
.. autofunction:: nanomesh.metrics.condition
.. autofunction:: nanomesh.metrics.distortion
.. autofunction:: nanomesh.metrics.max_angle
.. autofunction:: nanomesh.metrics.max_min_edge_ratio
.. autofunction:: nanomesh.metrics.min_angle
.. autofunction:: nanomesh.metrics.radius_ratio
.. autofunction:: nanomesh.metrics.relative_size_squared
.. autofunction:: nanomesh.metrics.scaled_jacobian
.. autofunction:: nanomesh.metrics.shape
.. autofunction:: nanomesh.metrics.shape_and_size

Reference
---------

The complete API reference is listed below.

.. toctree::
   :maxdepth: 2
   :caption: API reference
   :glob:

   nanomesh/*
