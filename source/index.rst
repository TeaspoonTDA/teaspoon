Welcome to teaspoon's documentation page!
==========================================

The emerging field of topological signal processing brings methods from Topological Data Analysis (TDA) to create new tools for signal processing by incorporating aspects of shape.
This python package, teaspoon for tsp or topological signal processing, brings together available software for computing persistent homology, the main workhorse of TDA, with modules that expand the functionality of teaspoon as a state-of-the-art topological signal processing tool.
These modules include methods for incorporating tools from machine learning, complex networks, information, and parameter selection along with a dynamical systems library to streamline the creation and benchmarking of new methods.
All code is open source with up to date documentation, making the code easy to use, in particular for signal processing experts with limited experience in topological methods.


Table of Contents
*******************

.. toctree::
   :maxdepth: 4
   :numbered:
   
   Getting Started <installation.rst>
   Modules <modules.rst>
   Contributing <contributing.rst>
   License <license.rst>
   Citing <citing.rst>



Collaborators
***************

The code is a compilation of work done by `Elizabeth Munch <http://www.elizabethmunch.com>`_ and `Firas Khasawneh <http://www.firaskhasawneh.com/>`_ along with her students and collaborators.  People who have contributed to teaspoon include:

- `Audun Myers <https://github.com/myersau3>`_
- Melih Yesilli
- `Sarah Tymochko <https://www.egr.msu.edu/~tymochko/>`_


Current Issues and To-do Items
*********************************************

This is a list of to do items:

	- Finish updating documentation
	- Make pip install-able
	- Validate (or make it so) that teaspoon can be imorted as "import teaspoon" and then functions can be called from there (e.g. teaspoon.MakeData.DynSysLib...).

This is a list of the current issues for Sphinx:

	- Get autodoc with sphinx to work. This works when Sarah removes a certain line in PD_classification file. I'm not sure which line this is.
	- There is no sphinx docstring documentation in the adaptivePart.py code.
	- Finish last few function documentations with examples.



Contact Information
********************
Liz Munch: muncheli@egr.msu.edu
