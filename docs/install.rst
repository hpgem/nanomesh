Installation
============

One of the goals for Nanomesh is that it is easy to install.
This means that all dependencies are available from `PyPi <https://pypi.org>`_.

If you use conda, it is advised to create a new environment:

::

   conda create -n nanomesh python=3.8
   conda activate nanomesh

Install nanomesh:

::

   pip install nanomesh

Note, `to enable the IPython
widgets <https://ipywidgets.readthedocs.io/en/latest/user_install.html#installation>`__:

::

   jupyter nbextension enable --py widgetsnbextension

Note, `if you are using Jupyter
lab <https://github.com/InsightSoftwareConsortium/itkwidgets#installation>`__:

::

   jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib jupyterlab-datawidgets itkwidgets

Tetgen
------

For tetrahedral meshing, Nanomesh requires `tetgen <https://wias-berlin.de/software/tetgen/>`__ to be
installed.

If you are a conda user, you can do:

::

    conda install -c conda-forge tetgen


Binaries are available `here <https://github.com/hpgem/tetgen/releases>`__, and compilation instructions `here <https://github.com/hpgem/tetgen/releases>`_.

Make sure `tetgen` is available on a directory on your system path. To verify tetgen is available, make sure that the following commands return a path:

Linux/MacOS

::

   which tetgen

Windows

::

   gcm tetgen.exe
