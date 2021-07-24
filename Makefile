# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = docs

# Put it first so that "make" without argument is like "make help".
# help:
	# @$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# .PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# Note: old way of passing in -M html results in docs going into the wrong folder.
	# For github pages to work, the docs have ot be in /docs/
	@$(SPHINXBUILD)  "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

	# echo "Running autopep8"
	@autopep8 -r --in-place teaspoon/
