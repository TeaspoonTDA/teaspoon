Getting Started
================


Requirements
**************

Please note that this code is an early version, so many things are not fully up and running yet.

Bugs reports and feature reqests can be posted on the `github issues page <https://github.com/lizliz/teaspoon/issues>`_.

In order to use all wrappers in teaspoon.TDA, the following need to be installed so they can be run command line.

- `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- `Perseus <http://people.maths.ox.ac.uk/nanda/perseus/index.html>`_. Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
- `Hera <https://bitbucket.org/grey_narn/hera>`_. Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.

<span style="color:red">Warning</span>: the teaspoon installation will not install nor check for the install of these packages. In particular, perseus should be installed so that typing `perseus` in a terminal runs it, even though the default perseus installation for some operating systems ends up with an executable with a different name.

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

Installation
**************

This package will soon be available through a pip install::

	pip install teaspoon


