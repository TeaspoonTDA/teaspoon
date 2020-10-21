Getting Started
================

Locations
**********

- **Code**: [https://github.com/lizliz/teaspoon](https://github.com/lizliz/teaspoon).

- **Documentation**: [http://elizabethmunch.com/code/teaspoon/index.html](http://elizabethmunch.com/code/teaspoon/index.html)


Requirements
**************

Please note that this code is still pre-alpha, so many things are not fully up and running yet.

Bugs reports and feature reqests can be posted on the [github issues](https://github.com/lizliz/teaspoon/issues) page.

In order to use all wrappers in teaspoon.TDA, the following need to be installed so they can be run command line.

- [Ripser](https://github.com/Ripser/ripser). Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/index.html). Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
- [Hera](https://bitbucket.org/grey_narn/hera). Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.

<span style="color:red">Warning</span>: the teaspoon installation will not install nor check for the install of these packages. In particular, perseus should be installed so that typing `perseus` in a terminal runs it, even though the default perseus installation for some operating systems ends up with an executable with a different name.



Installation
**************

This code is still pre-alpha.  In particular, using the pip install seems to be finicky at best.  
If you are having installation issues, please make a note of what you've done, including copying error message outputs, as a comment into the [installation issue on github](https://github.com/lizliz/teaspoon/issues) so we can start figuring out what is up with this system.


Installing using pip:
######################

So far, the most success has been had with pip installation.  Run the following commands to clone and install teaspoon. In particular, the `pip install .` command needs to be run in the folder with the `setup.py` file::

	git clone https://github.com/lizliz/teaspoon.git
	cd teaspoon
	pip install .


