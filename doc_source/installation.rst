Getting Started
================


Requirements
**************

Bugs reports and feature reqests can be posted on the `github issues page <https://github.com/TeaspoonTDA/teaspoon/issues>`_.

Most of the persistence computation is teaspoon is now done with `Scikit-TDA <https://scikit-tda.org/>`_, which is python based TDA computation. In a few legacy cases, the code still utilizes these other non-Python packages.

- `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- `Perseus <http://people.maths.ox.ac.uk/nanda/perseus/index.html>`_. Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.

**Required Dependencies:**

To install this package, both boost and CMake must be installed as system dependencies.  For boost see for `unix <https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html>`_ and `windows <https://www.boost.org/doc/libs/1_62_0/more/getting_started/windows.html>`_.  For mac, you can run ``brew install boost`` if using homebrew as a package manager. For CMake see `install CMake <https://cmake.org/install/>`_.

Installation
**************

The teaspoon package is available through pip install with version details found `here <https://pypi.org/project/teaspoon/>`_.
The package can be installed using the following pip installation:

	``pip install teaspoon``

To install the most up-to-date version of the code, you can clone the repo and then run::

  ``pip install .``
