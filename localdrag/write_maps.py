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
import cv2

import localdrag as ld

###--------------------------------------------------------------------------------

def write2txt(path, fn, prefix, array):
    """
    
    Parameters
    ----------
    path : string
        Path in storage to save file. 
    fn   : string
        Filename.
    prefix : string
        Prefix depending on map to write.
    array : numpy.ndarray
        Array/map to write.

    Returns
    -------
    Nothing.

    
    """
    fn_noPrefix = fn.replace("hx_", "")
    #np.savetxt(f'{path}/{prefix}_shape_{array.shape[0]}_{array.shape[1]}_{fn}.txt', np.flip(array, axis = 0))
    np.savetxt(f'{path}/{prefix}_{fn_noPrefix}.txt', np.flip(array, axis = 0))


def write2pgm(path, fn, array):
    """
    
    Parameters
    ----------
    path : string
        Path in storage to save file. 
    fn   : string
        Filename.
    array : numpy.ndarray
        Array/map to write.

    Returns
    -------
    Nothing.

    
    """
    #filename = f'{path}/{prefix}_shape_{array.shape[0]}_{array.shape[1]}_{fn}.pgm'
    filename = f'{path}/{fn}.pgm'
    
    # Scale array no range {0 - 255}
    array = np.around((array / np.max(array))  * 255)
    # Save as unsigned int8
    array = array.astype(np.uint8)
    
    cv2.imwrite(filename, array)


def write_2d_averaged_fields(path, fn, size, voxelsize, frames = []):
    """
    
    Parameters
    ----------
    path : string
        Path in storage to save file. 
    fn   : string
        Filename of the input/geometry file.
    prefix : string
        Prefix depending on map to write.
    size : [int, int, int]
        Size of the domain.
    voxelsize : float
        Voxelsize of the domain.
    frames : list of ints
        List of directions in which frames should be removed.

    Returns
    -------
    Nothing.
    
    Reads standart input from Stokes Solver simulations with fixed filenames.

    """
    if not os.path.isfile(fn):
        raise FileNotFoundError(f'Input file {fn} dose not exist!')

    # remove .raw form fn 
    filename = fn.replace('.raw', '')

    fields2d = ld.evaluate3d.get_2d_averaged_fields(path, fn, size, voxelsize, frames)

    if sorted(fields2d.keys()) == sorted(ld.constants.STOKES_FILE_PREFIXES):
        prefixs = ld.constants.STOKES_FILE_PREFIXES
        print(f'All files ({ld.constants.STOKES_FILE_PREFIXES}) available.')
    else:
        prefixs = fields2d.keys()
        print(f'Following files ({fields2d.keys()}) available.')

    # Write to txt
    for p in prefixs:
        write2txt(path, f'{filename}.txt', f'{p}2d', fields2d[p])


def remove_temp_frame(array):
    """
    
    Parameters
    ----------
    array : numpy.ndarray
        2d array 

    Returns
    -------
    array : numpy.ndarray
        Raw geometry without solid frame.
    array.shape : list [int, int]
        Size of array after removing frame. 
    
    Remove the temp solid frame in 2d
    pre-processing method.

    """
    
    array = array[1:array.shape[0]-1, :] 

    return array, array.shape




