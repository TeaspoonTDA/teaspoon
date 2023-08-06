# teaspoon.TDA

Code for using [Ripser](https://github.com/Ripser/ripser), in python.

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


