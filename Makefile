# You can set these variables from the command line, and also
# from the environment for the first two.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@Echo       autopep: runs autopep to improve formatting of code.
	@Echo       unittests: runs unit tests
	@Echo       clean_build: clears files from build folder
	@Echo       clean_docs: clears files from docs folder
	@Echo       html: makes documents in build folder
	@Echo       update_docs: Copy over contents of build/html to docs
	@Echo       all: runs clean build and docs folders, creates new html folders in build, moves relevant files to docs, and runs unittests and autopep.

autopep:
	# Running autopep8
	@autopep8 -r --in-place teaspoon/

unittests:
	# Running unittests
	@python -m unittest

clean_build:
	@mkdir $(shell pwd)/build/temp
	# Removing all files from build folder.
	@rm -rf $(shell pwd)/build/
	@mkdir $(shell pwd)/build/

clean_docs:
	@mkdir $(shell pwd)/docs/temp
	# Removing all files from build folder.
	@rm -rf $(shell pwd)/docs/
	@mkdir $(shell pwd)/docs/

html:
	# Running sphinx-build to build html files in build folder.
	sphinx-build -M html source build
	
update_docs:
	# Copying over contents of build/html to docs
	@mkdir $(shell pwd)/docs/.doctrees
	@cp -a $(shell pwd)/build/doctrees/. $(shell pwd)/docs/.doctrees/
	@cp -a $(shell pwd)/build/html/. $(shell pwd)/docs/

all:
	# Cleaning build folder
	@mkdir $(shell pwd)/build/temp
	@rm -rf $(shell pwd)/build/
	@mkdir $(shell pwd)/build/
	
	# Cleaning docs folder
	@mkdir $(shell pwd)/docs/temp
	@rm -rf $(shell pwd)/docs/
	@mkdir $(shell pwd)/docs/
	
	# Running sphinx-build to build html files.
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	
	# Copying over contents of build/html to docs
	@mkdir $(shell pwd)/docs/.doctrees
	@cp -a $(shell pwd)/build/doctrees/. $(shell pwd)/docs/.doctrees/
	@cp -a $(shell pwd)/build/html/. $(shell pwd)/docs/
	
	# Running autopep8
	@autopep8 -r --in-place teaspoon/
	
	# Running unittests
	@python -m unittest

	