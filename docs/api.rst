.. _api:

.. currentmodule:: nanomesh

Python Interface
================

This part of the documentation covers the public interface of Nanomesh.

The side bar contains a listing of classes and functions by topic.

.. toctree::
   :maxdepth: 1
   :hidden:

   api.image_data
   api.mesh_data
   api.meshing
   api.metrics
   api.helpers
   api.plotting
   api.data

Most of Nanomesh' functionality can be accessed through the classes
and functions listed below. See the :ref:`examples` for how to use them.

.. rubric:: Data types

.. autosummary::

   nanomesh.Plane
   nanomesh.Volume
   nanomesh.Image
   nanomesh.MeshContainer
   nanomesh.Mesh
   nanomesh.LineMesh
   nanomesh.TriangleMesh
   nanomesh.TetraMesh
   nanomesh.RegionMarker

.. rubric:: Meshing classes

.. autosummary::

   nanomesh.Mesher
   nanomesh.Mesher2D
   nanomesh.Mesher3D

.. rubric:: Functions

.. autosummary::

   nanomesh.simple_triangulate
   nanomesh.triangulate
   nanomesh.tetrahedralize
   nanomesh.volume2mesh
   nanomesh.plane2mesh

.. rubric:: Modules

.. autosummary::

   nanomesh.metrics
   nanomesh.data


Reference
=========

The complete API reference is listed below.

.. toctree::
   :maxdepth: 2
   :glob:

   nanomesh/*
