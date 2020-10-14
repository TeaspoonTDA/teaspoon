Description
==============



This package provides wrappers for using various persistent homology and other Topological Data Analysis software packages through python.


The code is an compilation of work done by [Elizabeth Munch](http://www.elizabethmunch.com) along with her students and collaborators.  People who have contributed to teaspoon include:


- [Firas Khasawneh](http://www.firaskhasawneh.com)
- Jesse Berwald
- Brian Bollen
- Audun Myers
- Melih Yesilli
- Sarah Tymochko


Locations
==============


- **Code**: [https://github.com/lizliz/teaspoon](https://github.com/lizliz/teaspoon).

- **Documentation**: [http://elizabethmunch.com/code/teaspoon/index.html](http://elizabethmunch.com/code/teaspoon/index.html)


Requirements
==============

Please note that this code is still pre-alpha, so many things are not fully up and running yet.

Bugs reports and feature reqests can be posted on the [github issues](https://github.com/lizliz/teaspoon/issues) page.

In order to use all wrappers in teaspoon.TDA, the following need to be installed so they can be run command line.

- [Ripser](https://github.com/Ripser/ripser). Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/index.html). Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
- [Hera](https://bitbucket.org/grey_narn/hera). Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.

<span style="color:red">Warning</span>: the teaspoon installation will not install nor check for the install of these packages. In particular, perseus should be installed so that typing `perseus` in a terminal runs it, even though the default perseus installation for some operating systems ends up with an executable with a different name.


Installation
==============

### <span style="color:red">Warning</span>:
This code is still pre-alpha.  In particular, using the pip install seems to be finicky at best.  
If you are having installation issues, please make a note of what you've done, including copying error message outputs, as a comment into the [installation issue on github](https://github.com/lizliz/teaspoon/issues) so we can start figuring out what is up with this system.



### Installing using pip:

So far, the most success has been had with pip installation.  Run the following commands to clone and install teaspoon. In particular, the `pip install .` command needs to be run in the folder with the `setup.py` file.

```{bash}
git clone https://github.com/lizliz/teaspoon.git
cd teaspoon
pip install .
```

### Installing using python:

According to the internet, the pip version of install appears to be better. However, if you don't use pip, another option is to cd into the teaspoon directory and run:

```{bash}
python setup.py install
```

### Alternatively, just add the folder to the python path.

If the above aren't working, add the following line to your `.bashrc` file.
```{bash}
export PYTHONPATH="${PYTHONPATH}:/path/to/teaspoon"
```

Documentation
=============

Documentation is done using [doxygen](http://www.doxygen.org).
The most recent version of the [teaspoon documentation is hosted here](http://elizabethmunch.com/code/teaspoon/index.html).  
Further info on documentation can be found in the [contributing](https://github.com/lizliz/teaspoon/blob/master/CONTRIBUTING.md) page.


Contributing
=================
See the [contributing](http://github.com/lizliz/teaspoon/blob/master/CONTRIBUTING.md) page for more information on workflows.

Contact
=============
Liz Munch: [muncheli@egr.msu.edu](mailto:muncheli@egr.msu.edu).
