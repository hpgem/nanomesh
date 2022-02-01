Development Notes
=================

Development Installation
------------------------

Install ``nanomesh`` using the development dependencies:

::

    conda create -n nanomesh-dev python=3.8
    conda activate nanomesh-dev

    pip install -e .[develop] -c constraints.txt

Running the tests:

::

    pytest

Linting/checks:

::

    pre-commit

Building the docs:

::

   make html --directory docs


Testing notebooks
-----------------

1. Install nanomesh kernel

   ::

       python -m ipykernel install --user --name nanomesh

2. Test notebooks

   ::

       cd notebooks
       pip install -r requirements.txt

   On Windows:

   ::

       ./test_notebooks.PS1

   On Linux/Mac:

   ::

       bash ./test_notebooks.sh


Making a release
----------------

1. Bump the version (major/minor/patch as needed)

    ::

        bumpversion minor

2. Make a new release. The github action to publish to pypi is triggered when a release is published.


Updating constraints.txt
------------------------

1. On Windows:
    - In a new environment

::

    pip freeze --exclude nanomesh > constraints.txt

2. On Linux:
    - In a new environment
    - Using the produced ``constraints.txt`` file

::

    pip install -e .[develop] -c constraints.txt
    pip freeze --exclude nanomesh >> constraints.txt
    sort --ignore-case constraints.txt | uniq > constraints_tmp.txt
    mv constraints_tmp.txt constraints.txt


Updating pre-commit
-------------------

::

    pre-commit autoupdate


Fixes for errors
----------------

If you get an error with pytest, like:

::

     from win32com.shell import shellcon, shell
    E   ImportError: DLL load failed while importing shell: The specified procedure could not be found.
    ImportError: DLL load failed while importing shell: The specified procedure could not be found.

Try:

::

    conda install -c conda-forge pywin32
