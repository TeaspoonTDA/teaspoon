Description
==============

The emerging field of topological signal processing brings methods from Topological Data Analysis (TDA) to create new tools for signal processing by incorporating aspects of shape.
This python package, teaspoon for tsp or topological signal processing, brings together available software for computing persistent homology, the main workhorse of TDA, with modules that expand the functionality of teaspoon as a state-of-the-art topological signal processing tool.
These modules include methods for incorporating tools from machine learning, complex networks, information, and parameter selection along with a dynamical systems library to streamline the creation and benchmarking of new methods.
All code is open source with up to date documentation, making the code easy to use, in particular for signal processing experts with limited experience in topological methods.


Full documentation of this package is available [here](https://teaspoontda.github.io/teaspoon/). The full documentation includes information about installation, module documentation with examples, contributing, the license, and citing teaspoon.

The code is a compilation of work done by [Elizabeth Munch](http://www.elizabethmunch.com) and [Firas Khasawneh](http://www.firaskhasawneh.com/) along with their students and collaborators.  People who have contributed to teaspoon include:

- [Audun Myers](https://www.audunmyers.com)
- [Melih Yesilli](https://www.melihcanyesilli.com)
- [Sarah Tymochko](https://www.egr.msu.edu/~tymochko/)
- [Danielle Barnes](https://github.com/barnesd8)
- [Ismail Guzel]

We gratefully acknowledge the support of the National Science Foundation, which has helped make this work possible.

Installation
=============
To install this package, both boost and CMake must be installed as system dependencies.  For boost see for [unix](https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html) and [windows](https://www.boost.org/doc/libs/1_62_0/more/getting_started/windows.html).  For mac, you can run ``brew install boost`` if using homebrew as a package manager.  For CMake see [here](https://cmake.org/install/).

The teaspoon package is available through pip install with version details found [here](https://pypi.org/project/teaspoon/).
The package can be installed using the following pip installation:

	``pip install teaspoon``

To install the most up-to-date version of the code, you can clone the repo and then run::

  ``pip install .``

from the main directory.  Note that the master branch will correspond to the version available in pypi, and the test_release branch may have new features.

Please reference the requirements page in the [documentation](https://teaspoontda.github.io/teaspoon/) for more details on other required installations.

Contacts
=============
* Liz Munch: [muncheli@msu.edu](mailto:muncheli@msu.edu).
* Firas Khasawneh: [khasawn3@egr.msu.edu](mailto:khasawn3@egr.msu.edu).
