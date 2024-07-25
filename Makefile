.PHONY: tests

help:
	@Echo       clean: runs autopep to improve formatting of code
	@Echo		reqs: replace requirements.txt file for use by collaborators to create virtual environments and for use in documentation
	@Echo       tests: runs unit tests
	@Echo       html: clears files from docs folder, rebuilds docs from source folder
	@Echo       release: runs build to create tar and wheel for distribution
	@Echo       all: runs clean build and docs folders, creates new html folders in build, moves relevant files to docs, and runs unittests and autopep.

reqs:
	@pip freeze > requirements.txt

clean:
	# Running autopep8
	@autopep8 -r --in-place teaspoon/

tests:
	# Running unittests
	@python -m unittest

release:
	python setup.py sdist bdist_wheel

html:
	# Running sphinx-build to build html files in build folder.
	rm -r docs
	mkdir docs
	sphinx-build -M html doc_source docs
	rsync -a docs/html/ docs/
	rm -r docs/html
	
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

	