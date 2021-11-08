# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXGEN     ?= sphinx-apidoc
SPHINXPROJ    = nanomesh
PYTHON        ?= python
PIP           ?= pip
SOURCEDIR     = .
BUILDDIR      = _build
TEMPLATESDIR  = _templates
STATICDIR     = _static

html: conf.py nanomesh.rst install.rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

nanomesh.rst:
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(SOURCEDIR)" ../nanomesh

examples.rst:
	$(PYTHON) make_examples.py $(notebooks)

install.rst:
	touch install.rst

clean:
	rm -rvf "$(BUILDDIR)" "$(STATICDIR)/examples" "$(STATICDIR)/banner.png"
	rm -rvf nanomesh*.rst examples.*rst
	rm -rvf apidocs/*.rst
