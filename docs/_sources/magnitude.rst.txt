Magnitude
=========

This module is related to calculation of the magnitude. Some relevant links and references are below. 

- `Magnitude in ncatlab <https://ncatlab.org/nlab/show/magnitude+of+an+enriched+category>`_.
- Tom Leinster, The magnitude of metric spaces. Doc. Math. 18 (2013), pp. 857-905. doi: `10.4171/DM/415 <https://doi.org/10.4171/DM/415>`_. 
- Miguel O'Malley, Sara Kalisnik, Nina Otter. Alpha magnitude. Journal of Pure and Applied Algebra, Volume 227, Issue 11,2023, doi: `10.1016/j.jpaa.2023.107396 <https://doi.org/10.1016/j.jpaa.2023.107396>`_.



Magnitude function
****************************************


.. automodule:: teaspoon.TDA.Magnitude
    :members: MagnitudeFunction
    :noindex:


.. This function calculates the magnitude function, ``t \\mapsto |tX|``, of an input distance matrix assumed to be calculated from a finite metric space ``X``, on the interval defined by `t_range` at `t_n` locations.

.. - Given a finite metric space ``(X,d)`` and for matrix purposes, fix an order on the points ``x_1,\\cdots,x_n``.
.. - Denote the distance matrix by ``D=[d(x_i,x_j)]_{ij}=[D_{ij}]_{ij}``
.. - Denote the similarity matrix ``Z=Z_X`` to have entries ``Z_{ij}=e^{-D_{ij}}``
.. - We'll also be interested in the scaled version for some ``t \\in (0,\\infty)``, where ``tZ`` is the matrix for metric space ``tX`` and ``tZ_{ij}=e^{-tD_{ij}}``
.. - The magnitude of ``|tX|`` is 
..   `` |tX| = \\sum_{i,j} ((tZ)^{-1})_{ij} ``
.. where ``(tZ)^{-1}`` is the inverse of the matrix ``tZ``, assuming it exists. 
.. - The magnitude function is 
..   ``M: t \\mapsto |tX|`` 
