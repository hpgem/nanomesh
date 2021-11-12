Installation
============

If you use conda, create a new environment:

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

For tetrahedral meshing, ``nanomesh`` requires `tetgen <https://wias-berlin.de/software/tetgen/>`__ to be
installed. See below for instructions on how to obtain/install
``tetgen`` on different platforms.


Compiling tetgen on Linux/Mac
-----------------------------

Follow the `compilation
instructions <https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual004.html#sec%3Acompile>`__.

Compiling tetgen on Windows
---------------------------

1. Install MinGW using `MSYS2 <https://www.msys2.org/>`__

2. Follow the installation instructions!

   -  Install/update packages:

      ::

          pacman -Syu

   -  Install development toolchain:

      ::

          pacman -S –needed base-devel mingw-w64-x86_64-toolchain

3. Add ``C:\msys64\mingw64\bin`` to the Windows Path environment
   variable. Click
   `here <https://code.visualstudio.com/docs/languages/cpp#_add-the-mingw-compiler-to-your-path>`__
   for info.

4. Compiling/installing tetgen:

   -  Compilation steps:

      ::

          g++ -O0 -c \predicates.cxx g++ -O3 -o tetgen
          tetgen.cxx predicates.o -lm

   -  Move ``tetgen.exe`` to a location on your system path
