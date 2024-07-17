#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Krach 
         david.krach@mib.uni-stuttgart.de

Copyright 2024 David Krach, Felix Weinhardt

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.

"""

### HEADER ------------------------------------------------------------------------

import numpy as np
import os
import scipy

###--------------------------------------------------------------------------------


def grad2d(array, vs):
    """
    
    Parameters
    ----------
    array : numpy.ndarray
        2D array.
    voxelsize : float
        Voxelsize of the domain.

    Returns
    -------
    array : numpy.ndarray
        Gradient field of array.
        Same shape as input domain.
    
    """
    
    return np.gradient(array, vs)


def laplace2d(array, voxelsize):
    """
    Parameters
    ----------
    array : numpy.ndarray
        2D array.
    voxelsize : float
        Voxelsize of the domain.

    Returns
    -------
    array : numpy.ndarray
        Divergence(Gradient(array)) field of array.
        Same shape as input domain.
    
    Compute the laplacian using `numpy.gradient` twice.
    Only for equidistante, regular grids.
    
    """
    
    grad_x, grad_y = np.gradient(array, voxelsize, voxelsize)
    grad_xx = np.gradient(grad_x, voxelsize, axis=0)
    grad_yy = np.gradient(grad_y, voxelsize, axis=1)
    
    return(grad_xx + grad_yy)


def h_squared(geom, voxelsize):
    """
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.
    voxelsize : float [ m ]
        Voxelsize of the domain.

    Returns
    -------
    h_sqared : np.ndarray
        h(x)**2
    
    """

    h_vox = geom.shape[0]
    h = np.zeros((geom.shape[1], geom.shape[2]), dtype = np.float64)
    for j in range(geom.shape[1]):
        for k in range(geom.shape[2]):
            h[j, k] = np.count_nonzero(geom[:, j, k]==0)/geom.shape[0]

    h = np.multiply(voxelsize * h_vox, h)
    return np.multiply(h, h)


def smooth_factor_map(a, sigma):
    """
    
    Parameters
    ----------
    array : numpy.ndarray
        2D array.
    sigma : float
        Standard deviation for Gaussian kernel.
    
    Returns
    -------
    array : numpy.ndarray
        Smoothed
    
    """

    return scipy.ndimage.gaussian_filter(a, sigma = sigma, mode = 'reflect')


def eigenvalues2d(array):
    """
    
    Parameters
    ----------
    array : 2 x 2 array 
        Second order permeability tensor.
    
    Returns
    -------
    Eigenvalues.
    """

    # check if there is no path one of the directions
    # if yes non-diagonal elements are zero
    if array[0, 1] == 0.0 or array[1, 0] == 0.0:
        a12 = 0.0
    else:
        a12 = (array[0, 1] + array[1, 0])/2.
    
    array[0, 1] = a12
    array[1, 0] = a12
    
    w, v = scipy.linalg.eig(np.asarray(array))


    if w.dtype == 'complex128' or w.dtype == 'complex64':
        return np.array( [w[0].real, w[1].real ] )
    else: 
        return np.asarray(w)


def eigendir_angle_2d(array, grad = True):
    """
    
    Parameters
    ----------
    array : 2 x 2 array 
        Second order permeability tensor..
    
    Returns
    -------
    Principal direction.
    """
    
    # check if there is no path one of the directions
    # if yes non-diagonal elements are zero
    if array[0, 1] == 0.0 or array[1, 0] == 0.0:
        a12 = 0.0
    else:
        a12 = (array[0, 1] + array[1, 0])/2.
    
    array[0, 1] = a12
    array[1, 0] = a12
    
    angle = 0.5 * np.arctan(2* array[0,1]/(array[0,0] - array[1,1]))
    if grad:
        return angle * 180./np.pi
    else:
        return angle
