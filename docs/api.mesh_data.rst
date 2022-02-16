.. _api_mesh_data:

.. currentmodule:: nanomesh

Working with mesh data
======================

A large part of Nanomesh deals with with generating and manipulating
2D and 3D meshes. To store the data, Nanomesh uses :class:`MeshContainer`,
which is a somewhat low-level, generic container for mesh data. It can
store different types of cells and associated data. It is used
to read/write data via `meshio <https://github.com/nschloe/meshio>`_.

To deal with mesh data more directly, use :class:`LineMesh`,
:class:`TriangleMesh` or :class:`TetraMesh`. These contain
dedicated methods for working with a specific type of mesh and plotting them.
Each can be generated from :class:`MeshContainer`.


.. rubric:: Classes

.. autosummary::

   nanomesh.MeshContainer
   nanomesh.LineMesh
   nanomesh.TriangleMesh
   nanomesh.TetraMesh


Reference
---------

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
