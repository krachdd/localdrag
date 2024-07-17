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
import matplotlib.pyplot as plt

###--------------------------------------------------------------------------------



def swap_0_1(array):
    """
    
    Parameters
    ----------
    array : numpy ndarray
        DESCRIPTION.
    
    Returns
    -------
    numpy ndarray.
    
    """
    
    array[array == 0]  = 99
    array[array == 1]  = 0
    array[array == 99] = 1
    
    return array


def eliminate_unconnected(array, binarized, zero_is_solid, verbose = False):
    """
    Parameters
    ----------
    array : numpy.ndarray
        array of the domain.
    binarized : bool
        If true, data is binarized 
    zero_is_solid : bool
        Flag what represents solid.
    verbose : bool, optional
        The default is True.

    Returns
    -------
    labeled_array : numpy.ndarray
        Same size as input array. 
        Unconnected fluid marked as solid.

    """
    
    if binarized == True and zero_is_solid == True:
        pass
    elif binarized == True and zero_is_solid == False:
        array = swap_0_1(array) # one is now fluid
    else:
        raise NotImplementedError('Not implemented for non binarized arrays!')

    labeled_array, num_features = scipy.ndimage.label(array)
    if verbose == True:
        print(f'{os.path.basename(__file__)}: Number of features: {num_features}')

    # get list of features in first and last slide
    features_first_slide = np.unique(labeled_array[:, 0])
    features_last_slide  = np.unique(labeled_array[:, -1])

    # common features, remove 0 since solid anyhow
    common_features = np.intersect1d(features_first_slide, features_last_slide)
    if np.all(common_features) == False:
        common_features = np.delete(common_features, 0)
    
    if verbose == True: 
        print(f'{os.path.basename(__file__)}: Number of common features: {common_features.shape[0]}')

    labeled_array[labeled_array == 0] = 255
    
    # all voxels on percolating paths set to 0
    for i in common_features:
        labeled_array[labeled_array == i] = 0

    # all disconnected pores are set to 1 if not part of the non connected features
    labeled_array[labeled_array != 0] = 1
    labeled_array[labeled_array == 255] = 1

    if binarized == True and zero_is_solid == True:
        labeled_array = swap_0_1(labeled_array)
    elif binarized == True and zero_is_solid == False:
        pass
    else:
        raise NotImplementedError('Not implemented for non binarized arrays!')

    return labeled_array


def mirror(array, verbose = False):
    """
    Parameters
    ----------
    array : numpy.ndarray
        array of the domain.

    Returns
    -------
    labeled_array : numpy.ndarray
        Mirrored in both lateral directions of microfluidic chip.
    """

    if verbose == True:
        print(f'{os.path.basename(__file__)}: Initial size: {array.shape}')
    g0 = np.flip(array, 0)
    array = np.concatenate((array, g0), axis = 0)
    g1 = np.flip(array, 1)
    array = np.concatenate((array, g1), axis = 1)
    if verbose == True:
        print(f'{os.path.basename(__file__)}: Returned size: {array.shape}')
    
    return array

def porosity(array):
    """
    Parameters
    ----------
    array : numpy.ndarray
        array of the domain.

    Returns
    -------
    porosity : float
    
    """

    return np.sum(array)/(array.shape[0]*array.shape[1])