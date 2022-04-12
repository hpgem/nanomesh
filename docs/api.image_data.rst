.. _api_image_data:

.. currentmodule:: nanomesh

Image data
==========

Nanomesh has two classes for representing image data, :class:`Plane` for working
with 2D pixel data and :class:`Volume` for working with 3D voxel data.
These can be used to load, crop, transform, filter, and segment image data.

Both classes derive from :class:`Image`. Instantiating :class:`Image` will
create the appropriate subclass.


.. rubric:: Classes

.. autosummary::

   nanomesh.Image
   nanomesh.Plane
   nanomesh.Volume

Reference
---------

.. autoclass:: Image
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

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
