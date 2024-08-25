Getting Started
================




Installation
**************

The teaspoon package is available through pip install with version details found `here <https://pypi.org/project/teaspoon/>`_.
The package can be installed using the following pip installation::

	pip install teaspoon

To install the most up-to-date version of the code, you can clone the repo and then run::

  pip install .

Optional Dependencies
*********************

To the greatest extent possible, we are trying to make install as simple as possible. However, the following optional dependencies are required for some of the modules in teaspoon, but due to issues with direct pip install, they are not included automatically when installing teaspoon.  

#. `Fast-zigzag software (fzz) <https://github.com/TDA-Jyamiti/fzz>`_ is required for the ZigZag persistence module. The software can be installed by following the instructions on the GitHub page. The software is not available through pip install.
#. In a few legacy cases, the code utilizes `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix. However, the majority of persistence computation is done with `Scikit-TDA <https://scikit-tda.org/>`_.


.. Spork!
.. **************

.. For our long time teaspoon users, you might notice the install for teaspoon has been simplified.  Based on user feedback, we've separated out modules which require boost and CMake as system dependencies.  
.. To use the modules including ZigZag persistence, please visit the teaspoon companion package, `spork <https://teaspoontda.github.io/spork/index.html>`_.

Issues
******

Bugs reports and feature reqests can be posted on the `github issues page <https://github.com/TeaspoonTDA/teaspoon/issues>`_.

