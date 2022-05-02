Nanomesh documentation
======================

|Documentation Status| |tests| |PyPI - Python Version| |PyPI| |DOI|

.. figure:: _static/banner.png
   :alt: Nanomesh banner

Welcome to the nanomesh documentation!

Nanomesh is a Python workflow tool for generating meshes from 2D and 3D image data. It has an easy-to-use API that can help process and segment image data, generate quality meshes (triangle / tetrahedra), and write the data to many mesh formats. Nanomesh also contains tools to inspect the meshes, visualize them, and generate cell quality metrics.

- Easy-to-use Python API
- Segment and mesh 2D or 3D image data
- Mesh visualization
- Calculate and plot cell metrics
- Export to many mesh formats

Try nanomesh in your browser!
-----------------------------

.. list-table::

   * - .. figure:: _static/meshing_dash.png
          :alt: Image of meshing dashboard
          :target: https://share.streamlit.io/hpgem/nanomesh/update_dash/dash/meshing_dash.py
          :align: center
          :width: 85%

          `Generate a 2D mesh <https://share.streamlit.io/hpgem/nanomesh/update_dash/dash/meshing_dash.py>`_

     - .. figure:: _static/metrics_dash.png
          :alt: Image of metrics dashboard
          :target: https://share.streamlit.io/hpgem/nanomesh/update_dash/dash/metrics_dash.py
          :align: center
          :width: 85%

          `Calculate mesh metrics <https://share.streamlit.io/hpgem/nanomesh/update_dash/dash/metrics_dash.py>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   development
   examples/other_hello_world!
   structure


.. toctree::
   :maxdepth: 1
   :caption: How-to's

   examples/index


.. toctree::
   :maxdepth: 1
   :caption: API

   api.rst


.. toctree::
   :caption: Links

   👨‍💻 Source code <https://github.com/HPGEM/nanomesh>
   💡 Issues <https://github.com/HPGEM/nanomesh/issues>
   📢 Releases <https://github.com/hpgem/nanomesh/releases>
   🐍 PyPI <https://pypi.org/project/nanomesh>
   📚 documentation <https://nanomesh.readthedocs.io>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |Documentation Status| image:: https://readthedocs.org/projects/nanomesh/badge/?version=latest
   :target: https://nanomesh.readthedocs.io/en/latest/?badge=latest
.. |tests| image:: https://github.com/hpgem/nanomesh/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/hpgem/nanomesh/actions/workflows/test.yaml
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/nanomesh
   :target: https://pypi.org/project/nanomesh/
.. |PyPI| image:: https://img.shields.io/pypi/v/nanomesh.svg?style=flat
   :target: https://pypi.org/project/nanomesh/
.. |DOI| image:: https://zenodo.org/badge/311460276.svg
   :target: https://zenodo.org/badge/latestdoi/311460276
