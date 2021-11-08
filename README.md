[![Documentation Status](https://readthedocs.org/projects/nanomesh/badge/?version=latest)](https://nanomesh.readthedocs.io/en/latest/?badge=latest)
[![Linux](https://github.com/hpgem/nanomesh/actions/workflows/test_on_linux.yml/badge.svg)](https://github.com/hpgem/nanomesh/actions/workflows/test_on_linux.yml)
[![MacOS](https://github.com/hpgem/nanomesh/actions/workflows/test_on_macos.yaml/badge.svg)](https://github.com/hpgem/nanomesh/actions/workflows/test_on_macos.yaml)
[![Windows](https://github.com/hpgem/nanomesh/actions/workflows/test_on_windows.yaml/badge.svg)](https://github.com/hpgem/nanomesh/actions/workflows/test_on_windows.yaml)

![Nanomesh banner](./notebooks/banner/banner.png)

# nanomesh

Creates 3d meshes from microscopy experimental data.

Documentation: https://nanomesh.readthedocs.io/en/latest/

## Installation

If you use conda, create a new environment:

```
conda create -n nanomesh python=3.8
conda activate nanomesh
```

Install nanomesh:

```
pip install nanomesh
```

For the full installation instructions, see the [documentation](https://nanomesh.readthedocs.io/en/latest/).

### Development

Install `nanomesh` using the development dependencies:

`pip install -e .[develop] -c constraints.txt`

Running the tests:

`pytest`

Linting/checks:

`pre-commit`

Building the docs:

```
make html --directory docs
```
