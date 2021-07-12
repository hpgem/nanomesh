import sys
from pathlib import Path

root = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(root))

DOCSDIR = '.'
BUILDDIR = '_build'
TEMPLATESDIR = '_templates'
STATICDIR = '_static'
SOURCEDIR = '../nanomesh'


# -- Run apidoc plug-in manually, as readthedocs doesn't support it -------
# See https://github.com/rtfd/readthedocs.org/issues/1139
def run_apidoc(app):
    ignore_paths = []

    cmd = ('--separate --no-toc --force --module-first '
           f'-t {TEMPLATESDIR} '
           f'-o {DOCSDIR} '
           f'{SOURCEDIR} ')

    args = cmd.split() + ignore_paths

    from sphinx.ext import apidoc
    apidoc.main(args)


# Convert readme.md to rst to be included in index.html
def make_readme(app):
    import subprocess
    cmd = 'pandoc --from=markdown --to=rst --output=README.rst ../README.md'
    args = cmd.split()
    subprocess.run(args)


# https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
# https://github.com/readthedocs/readthedocs.org/issues/2276
def setup(app):
    app.connect('builder-inited', make_readme)
    app.connect('builder-inited', run_apidoc)


extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'nbsphinx_link',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.viewcode',
    # 'autodocsumm',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'nanomesh'
copyright = u'2020, '
author = u'Nicolas Renaud'

# The short X.Y version.
version = release = '0.1.0'

# The language for content autogenerated by Sphinx.
language = 'english'

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

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
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

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

autodoc_mock_imports = [
    'ipywidgets',
    'itkwidgets',
    'pygalmesh',
    'open3d',
    'pyvista',
    'trimesh',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}
