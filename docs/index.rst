.. nanomesh documentation master file, created by
   sphinx-quickstart on Thu Jun 21 11:07:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|Documentation Status| |Linux| |MacOS| |Windows|

.. figure:: _static/banner.png
   :alt: Nanomesh banner


Welcome to the nanomesh documentation!

Nanomesh is a python workflow tool for generating meshes from 2D and 3D microscopy image data.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   development


.. toctree::
   :maxdepth: 1
   :caption: Examples
   :glob:

   examples/*


.. toctree::
   :maxdepth: 1
   :caption: API reference

   nanomesh.rst
   nanomesh.image.rst
   nanomesh.image2mesh.rst
   nanomesh.mesh.rst
   nanomesh.io.rst
   nanomesh.mesh_container.rst
   nanomesh.metrics.rst
   nanomesh.plotting.rst
   nanomesh.utils.rst


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
