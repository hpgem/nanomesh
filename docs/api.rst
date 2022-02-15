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
.. autoclass:: LineMesh
.. autoclass:: TriangleMesh
.. autoclass:: TetraMesh

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

Reference
---------

The complete API reference is listed below.

.. toctree::
   :maxdepth: 2
   :caption: API reference
   :glob:

   nanomesh/*
