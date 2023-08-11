Getting Started
================


Requirements
**************

Bugs reports and feature reqests can be posted on the `github issues page <https://github.com/TeaspoonTDA/teaspoon/issues>`_.

Most of the persistence computation is teaspoon is now done with `Scikit-TDA <https://scikit-tda.org/>`_, which is python based TDA computation. In a few legacy cases, the code still utilizes these other non-Python packages.

- `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.

Installation
**************

The teaspoon package is available through pip install with version details found `here <https://pypi.org/project/teaspoon/>`_.
The package can be installed using the following pip installation:

	``pip install teaspoon``

To install the most up-to-date version of the code, you can clone the repo and then run::

  ``pip install .``

Spork!
**************

For our long time teaspoon users, you might notice the install for teaspoon has been simplified.  Based on user feedback, we've separated out modules which require boost and CMake as system dependencies.  
To use the modules including ZigZag persistence, please visit the teaspoon companion package, `spork <https://teaspoontda.github.io/spork/index.html>`_.