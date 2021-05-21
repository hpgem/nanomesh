[![Documentation Status](https://readthedocs.org/projects/nanomesh/badge/?version=latest)](https://nanomesh.readthedocs.io/en/latest/?badge=latest)

# nanomesh

Creates 3d meshes from electron microscopy experimental data

The project setup is documented in [a separate document](project_setup.rst). Feel free to remove this document (and/or
the link to this document) if you don\'t need it.

## Installation

First install the dependencies for pygalmesh (CGAL and EIGEN3)

```
sudo apt-get install libcgal-dev libeigen3
```

If you use conda, create a new environment:

```
conda create -n nanomesh python=3.8
conda activate nanomesh
```

Install nanomesh:

```
git clone https://github.com/hpgem/nanomesh.git
cd nanomesh
pip install .
```

Note: To enable the IPython widgets:

```
jupyter nbextension enable --py widgetsnbextension
```

Note: If you are using Jupyter lab:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib jupyterlab-datawidgets itkwidgets
```

## Development

Running the tests:

`pytest`

Linting/checks:

`pre-commit`

Building and viewing the docs:

```
make html --directory docs
sensible-browser docs/_build/html/index.html
```


### License

Copyright (c) 2020,

Licensed under the Apache License, Version 2.0 (the \"License\"); you
may not use this file except in compliance with the License. You may
obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an \"AS IS\" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

### Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[NLeSC/python-template](https://github.com/NLeSC/python-template).
