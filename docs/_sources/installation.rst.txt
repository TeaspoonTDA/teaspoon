Getting Started
================


Requirements
**************

Please note that this code is an early version, so many things are not fully up and running yet.

Bugs reports and feature reqests can be posted on the `github issues page <https://github.com/lizliz/teaspoon/issues>`_.

Most of the persistence computation is teaspoon is now done with `Scikit-TDA <https://scikit-tda.org/>`_, which is python based TDA computation. In a few legacy cases, the code still utilizes these other non-Python packages.

- `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- `Perseus <http://people.maths.ox.ac.uk/nanda/perseus/index.html>`_. Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
- `Hera <https://bitbucket.org/grey_narn/hera>`_. Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.

**Warning:** The teaspoon installation will not install nor check for the install of these packages. In particular, perseus should be installed so that typing `perseus` in a terminal runs it, even though the default perseus installation for some operating systems ends up with an executable with a different name.

**Required Packages:**

* pip install pyentrp
* pip install numpy
* pip install matplotlib
* pip install os
* pip install sys
* pip install itertools
* pip install sci-kit
* pip install networkx
* pip install scipy
* pip install POT
* pip install persim
* pip install --verbose dionysus

Installation
**************

This package is available through a pip install::

	pip install teaspoon
