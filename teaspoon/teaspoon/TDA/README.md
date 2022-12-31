# teaspoon.TDA

Code for using [Ripser](https://github.com/Ripser/ripser), [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/index.html), and [Hera](https://bitbucket.org/grey_narn/hera) in python.

## General notes on format

All diagrams are stored as an Lx2 numpy matrix.
Code for computing persistence returns a dictionary with multiple dimensions of persistence where keys are the dimensions.
```python
Dgms = {
    0: DgmDimension0,
    1: DgmDimension1,
    2: DgmDimension2
}
```
Infinite classes are given an entry of np.inf.
*TODO: Check that this is true.*

In the process of computation, data files are saved in a hidden folder .teaspoonData
This folder is created if it doesn't already exist.
Files are repeatedly emptied out of it, so do not save anything in it that you might want later!

----

## Persistence for Point Clouds

### Using Ripser

To date, this is the fastest code for doing point cloud persistence. 
Computes persistence of a point cloud using [Ripser](https://github.com/Ripser/ripser).

```{python}
VR_Ripser(P, maxDim = 1)
```


Note: This code actually just computes the pairwise distance matrix and passes it to distMat_Ripser


#### Parameters
- P = 
    A point cloud as an NxD numpy array.
    N is the number of points, D is the dimension of
    Euclidean space.
- maxDim = 
    An integer representing the maximum dimension
    for computing persistent homology.

#### Returns

- Dgms = 
    A dictionary where Dgms[k] is an Lx2 matrix, where L is the number of points in the persistence diagram.  Infinite classes are given with an np.inf entry.
        
        
### Using Perseus

```python
VR_Perseus(P,dim = 1, 
            maxRadius = 3, numSteps = 100, stepSize = None,
            suppressOutput = True)
```

Does brips version of perseus.
Computes VR persitsence on points in Euclidean space.

#### Warnings:

1. Requires choice of maxRadius, numSteps, and/or stepSize. 
   Bad choices will give junk results. So, you'll need at least a little knowledge of your data a priori.
2. TODO: Perseus appears to spit out radius rather than diameter
   persistence computations.  Need to figure this out and 
   make the choice uniform across outputs.


#### Parameters

- P  
    - An NxD array.  Represents N points in R^D.
- maxRadius, stepSize, numSteps  
    - Only 2 of the three entries should be passed. 
    - Perseus requires that you decide how many steps, and how wide they are, rather than computing all possible topological changes.  So, persistence will be calculated from parameter 0 until  
        maxRadius = stepSize*numSteps.
 
    - If numSteps and stepSize are passed (regardless of whether maxRadius is passed), they will be used for the computation.  Otherwise, the two non-none valued entries will be used to calculate the third.
    *TODO: Check that this is actually the behavior!*

- suppressOutput
    - If true, gets rid of printed output from perseus.

#### Outputs

- Dgms
    - A dictionary with integer keys 0,1,...,N 
    The key gives the dimension of the persistence diagram.


---

## Persistence for Distance Matrices

Computes persistence of data given as a pairwise distance matrix using [Ripser](https://github.com/Ripser/ripser).

```python
distMat_Ripser(distMat, maxDim = 1)
```

#### Parameters
- distMat
    - A pairwise distance matrix (note: symmetric!) given as an NxN numpy array.
- maxDim
    - An integer representing the maximum dimension for computing persistent homology.

#### Returns

- Dgms = 
    A dictionary where Dgms[k] is an Lx2 matrix, where L is the number of points in the persistence diagram.  Infinite classes are given with an np.inf entry.        



---

## Persistence for Cubical Complexes (Images)

Computes persistence for a matrix of function values.
Uses Vidit Nanda's perseus.

```python
Cubical_Perseus(M, numDigits = 2, suppressOutput = True):
```

**Warnings:**

1. Perseus must be in the bash path
2. Matrix must be 2-dimensional.  *TODO: Update this to accept higher dimensional cubical complexes.*

#### Parameters:

- M  
    - A 2D numpy array representing the image.
- numDigits
    - Perseus only accepts positive integer valued matrices.  To 
    compensate, we apply the transformation  
            `x -> x* (10**numDigits) + M.min()  `  
    then calculate persistence on the resulting matrix.
    The persistence diagram birth/death times are then converted
    back via the inverse transform.
- suppressOutput
    - If true, gets rid of printed output from perseus.

#### Outputs

- Dgms
    - A dictionary with integer keys 0,1,...,N. 
        The key gives the dimension of the persistence diagram.

----

