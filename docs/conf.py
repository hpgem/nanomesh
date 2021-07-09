# -*- coding: utf-8 -*-
#
# nanomesh documentation build configuration file, created by
# sphinx-quickstart on Mon Nov 09 09:52:54 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

root = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(root))

DOCSDIR = '.'
BUILDDIR = '_build'
TEMPLATESDIR = '_templates'
STATICDIR = '_static'
SOURCEDIR = '../nanomesh'


def make_examples(app):
    import make_examples
    make_examples.main()


def make_readme(app):
    import subprocess
    cmd = 'pandoc --from=gfm --to=rst --output=README.rst ../README.md'
    args = cmd.split()
    subprocess.run(args)


def make_apidoc(app):
    import subprocess
    cmd = 'sphinx-apidoc -eTf -t {TEMPLATESDIR} -o {DOCSDIR} {SOURCEDIR}'
    args = cmd.split()
    subprocess.run(args)


def setup(app):
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    # https://github.com/readthedocs/readthedocs.org/issues/2276
    app.connect('builder-inited', make_examples)
    app.connect('builder-inited', make_readme)
    app.connect('builder-inited', make_apidoc)


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.viewcode',
    # 'autodocsumm',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'nanomesh'
copyright = u'2020, '
author = u'Nicolas Renaud'

# The short X.Y version.
version = release = '0.1.0'

# The language for content autogenerated by Sphinx.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [BUILDDIR, 'Thumbs.db', '.DS_Store']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [STATICDIR]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Use autoapi.extension to run sphinx-apidoc
autoapi_dirs = [SOURCEDIR]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme_options = {}

# options for rtd-theme
# html_theme_options = {
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     # toc options
#     'collapse_navigation': True,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False,
# }

autodoc_default_options = {
    'autosummary': True,
    'special-members': '__init__',
}

autodoc_mock_imports = [
    'ipywidgets',
    'itkwidgets',
    'pygalmesh',
    'open3d',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}
