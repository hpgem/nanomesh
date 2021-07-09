# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXGEN     ?= sphinx-apidoc
SPHINXPROJ    = nanomesh
NBCONVERT     ?= jupyter
PYTHON        ?= python
PIP           ?= pip
SOURCEDIR     = .
BUILDDIR      = _build
TEMPLATESDIR  = _templates
STATICDIR     = _static

# ../notebooks/logo/logo.ipynb \

notebooks = \
../notebooks/sample_data/multi-domain-plane.ipynb \
../notebooks/sample_data/process_raw_data.ipynb \
../notebooks/sample_data/sample_data.ipynb \
../notebooks/sample_data/sample_data_2D.ipynb \
../notebooks/sample_data/sample_data_3D.ipynb \
../notebooks/sample_data/select_roi_2d.ipynb

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: conf.py nanomesh.rst README.rst examples.rst install.rst examples.rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	mv "$(BUILDDIR)/html/_static/examples" "$(BUILDDIR)/html/examples" || true

README.rst: ../README.md
	pandoc --from=gfm --to=rst --output=README.rst ../README.md

nanomesh.rst:
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(SOURCEDIR)" ../nanomesh

examples.rst:
	$(PYTHON) make_examples.py $(notebooks)

install.rst:
	touch install.rst

clean:
	rm -rvf "$(BUILDDIR)" "$(STATICDIR)/examples" nanomesh*.rst README.rst examples.*rst
