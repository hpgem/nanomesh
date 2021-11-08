Development Notes
=================

Development Installation
------------------------

Install ``nanomesh`` using the development dependencies:

::

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
