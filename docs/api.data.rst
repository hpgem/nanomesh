.. _api_data:

.. module:: nanomesh.data

Sample Data
===========

The :mod:`nanomesh.data` module helps provides some standard data sets to work with.

.. rubric:: Image data

These image data are currently available in Nanomesh.

.. autosummary::

   binary_blobs2d
   binary_blobs3d
   nanopores
   nanopores3d
   nanopores_gradient

.. seealso::

    For additional image data sets, have a look at :mod:`skimage.data`:

   - :func:`skimage.data.cells3d`
   - :func:`skimage.data.horse`
   - :func:`skimage.data.binary_blobs`
   - :func:`skimage.data.coins`

.. rubric:: Mesh data

These mesh data are currently available.

.. autosummary::

   blob_mesh2d
   blob_mesh3d

Reference
---------

.. autofunction:: binary_blobs2d
.. autofunction:: binary_blobs3d
.. autofunction:: blob_mesh2d
.. autofunction:: blob_mesh3d
.. autofunction:: nanopores
.. autofunction:: nanopores3d
.. autofunction:: nanopores_gradient
