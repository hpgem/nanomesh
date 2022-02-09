.. nanomesh documentation master file, created by
   sphinx-quickstart on Thu Jun 21 11:07:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|Documentation Status| |Linux| |MacOS| |Windows|

.. figure:: _static/banner.png
   :alt: Nanomesh banner


Welcome to the nanomesh documentation!


Getting started
===============
.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   development


Examples
========
.. toctree::
   :maxdepth: 1
   :caption: Examples
   :glob:

   examples/*


API Reference
=============

.. toctree::
   :maxdepth: 1
   :caption: API reference

   nanomesh.io.rst
   nanomesh.mesh2d.rst
   nanomesh.mesh3d.rst
   nanomesh.mesh_container.rst
   nanomesh.mesh_utils.rst
   nanomesh.mesh_utils_3d.rst
   nanomesh.metrics.rst
   nanomesh.image.rst
   nanomesh.roi2d.rst
   nanomesh.tetgen.rst
   nanomesh.utils.rst


Links
=====
.. toctree::
   :caption: Links

   🔗 Source code <http://github.com/HPGEM/nanomesh>
   🔗 Issues <http://github.com/HPGEM/nanomesh/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |Documentation Status| image:: https://readthedocs.org/projects/nanomesh/badge/?version=latest
   :target: https://nanomesh.readthedocs.io/en/latest/?badge=latest
.. |Linux| image:: https://github.com/hpgem/nanomesh/actions/workflows/test_on_linux.yml/badge.svg
   :target: https://github.com/hpgem/nanomesh/actions/workflows/test_on_linux.yml
.. |MacOS| image:: https://github.com/hpgem/nanomesh/actions/workflows/test_on_macos.yaml/badge.svg
   :target: https://github.com/hpgem/nanomesh/actions/workflows/test_on_macos.yaml
.. |Windows| image:: https://github.com/hpgem/nanomesh/actions/workflows/test_on_windows.yaml/badge.svg
   :target: https://github.com/hpgem/nanomesh/actions/workflows/test_on_windows.yaml
