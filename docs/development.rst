.. _development:

Development Notes
=================

Development Installation
------------------------

Nanomesh does not have any hard version constraints. For development, it is
still useful to have a consistent environment.

Therefore, Nanomesh uses a constraints file (``constraints.txt``) which pins the version requirements.

The constraints are automatically `updated and tested every month <https://github.com/hpgem/nanomesh/actions/workflows/update_dependencies.yaml>`__.

Note that there is a `Github action <https://github.com/hpgem/nanomesh/actions/workflows/update_dependencies.yaml>`__ to update and test the dependencies every month.

In case you run into issues, you may also try to install
Nanomesh with constraints file.

Install Nanomesh using the development dependencies:

::

    conda create -n nanomesh-dev python=3.8
    conda activate nanomesh-dev

    pip install -e .[develop] -c constraints.txt

Running the tests using `pytest <https://docs.pytest.org/>`__:

::

    pytest

To check coverage:

::

    coverage -m run pytest
    coverage report  # terminal
    coverage html    # html report

Linting and checks are done using `pre-commit <https://pre-commit.com>`__:

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
