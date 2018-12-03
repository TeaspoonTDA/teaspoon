.. teaspoon documentation master file, created by
   sphinx-quickstart on Thu Sep 27 16:51:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to teaspoon's documentation!
====================================

.. image:: teaspoon.png


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   teaspoon/relatedpages
   teaspoon/packages
   teaspoon/classes


Description
==============



This package provides wrappers for using various persistent homology and other Topological Data Analysis software packages through python.


The code is an compilation of work done by `Elizabeth Munch <http://www.elizabethmunch.com>`_ along with her students and collaborators.  People who have contributed to teaspoon include:


- `Firas Khasawneh <http://www.firaskhasawneh.com>`_
- Jesse Berwald
- Brian Bollen


Locations
==============


- **Code**: `http://gitlab.msu.edu/TSAwithTDA/teaspoon <https://gitlab.msu.edu/TSAwithTDA/teaspoon>`_.

- **Documentation**: `http://elizabethmunch.com/code/teaspoon/index.html <http://elizabethmunch.com/code/teaspoon/index.html>`_


Requirements
==============

Please note that this code is still pre-alpha, so many things are not fully up and running yet.

Bugs reports and feature requests can be posted on the `gitlab issues <https://gitlab.msu.edu/TSAwithTDA/teaspoon/issues>`_ page.

In order to use all wrappers in teaspoon.TDA, the following need to be installed so they can be run command line.

- `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- `Perseus <http://people.maths.ox.ac.uk/nanda/perseus/index.html>`_. Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
- `Hera <https://bitbucket.org/grey_narn/hera>`_. Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.

.. warning::

  the teaspoon installation will not install nor check for the install of these packages. In particular, perseus should be installed so that typing `perseus` in a terminal runs it, even though the default perseus installation for some operating systems ends up with an executable with a different name.


Installation
==============

.. warning::

  This code is still pre-alpha.  In particular, using the pip install seems to be finicky at best.

If you are having installation issues, please make a note of what you've done, including copying error message outputs, as a comment into the `installation issue on gitlab <https://gitlab.msu.edu/TSAwithTDA/teaspoon/issues/1>`_ so we can start figuring out what is up with this system.



**Installing using pip**:

So far, the most success has been had with pip installation.  Run the following commands to clone and install teaspoon. In particular, the `pip install .` command needs to be run in the folder with the `setup.py` file.::

  git clone https://gitlab.msu.edu/TSAwithTDA/teaspoon
  cd teaspoon
  pip install .


**Installing using python**:

According to the internet, the pip version of install appears to be better. However, if you don't use pip, another option is to cd into the teaspoon directory and run::

  python setup.py install


**Alternatively, just add the folder to the python path**.

If the above aren't working, add the following line to your `.bashrc` file.::

  export PYTHONPATH="${PYTHONPATH}:/path/to/teaspoon"


Documentation
=============

Documentation is done using `doxygen <http://www.doxygen.org>`_.
The most recent version of the `teaspoon documentation is hosted here <http://elizabethmunch.com/code/teaspoon/index.html>`_.
Further info on documentation can be found in the `contributing <https://gitlab.msu.edu/TSAwithTDA/teaspoon/blob/master/CONTRIBUTING.md>`_ page.


Contributing
=============

See the `contributing <https://gitlab.msu.edu/TSAwithTDA/teaspoon/blob/master/CONTRIBUTING.md>`_ page for more information on workflows.

Contact
=============
Liz Munch: `muncheli@egr.msu.edu <mailto:muncheli@egr.msu.edu>`_.
