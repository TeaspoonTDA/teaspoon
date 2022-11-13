Contributing to Teaspoon
=======================================================

Contributions are more than welcome! There are lots of opportunities for potential projects, so please get in touch if you would like to help out. Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.

Running the makefile will also recursively run autopep8 on all files.

Further checks for clean code can by run with
```
pylint --rcfile=pylintrc path/to/file.py
```
using the pylintrc at the top level folder in order to specify camelCase.

Contributing to Documentation
*******************************

If contributing to the documentation, the following python packages need to be pip installed:

	- pip install persim
	- pip install sphinxcontrib-bibtex
	- pip install sphinx-rtd-theme
	- pip install sphinx-prompt

Assuming your sphinx installation is up-to-date, you can run `make` from the teaspoon directory to update the documentation. Note that github pages requires that the documentation is in a folder named `docs`.
