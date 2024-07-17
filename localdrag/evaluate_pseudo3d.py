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
import os, glob, sys
import matplotlib.pyplot as plt
import scipy
import cv2

import localdrag as ld

###--------------------------------------------------------------------------------




def pseudo3d_vtu2txt(path, prefix):
    """
    Parameters
    ----------
    array : np.ndarray
        Collapse arrays first axis by assigning mean value
        
    
    Returns
    -------
    2d array with shape [array.shape[1], array.shape[2]]
    
    """

    script_dir = os.getcwd()
    os.chdir(path)
    files = glob.glob('*01.vtu')
    
    for filename in files:
        data = ld.wrap_import.openvtu_2d_dumux(filename, constants.DUMUX_FILE_PREFIXES)

        points, velx, vely, velmag, p, h = ld.evaluate_pseudo3d.extract_fields(data) 
        vs = ld.wrap_import.getVoxelSizeFromName(filename)
        size = ld.wrap_import.getNumVoxelFromName(filename)
        xsize = size[1]
        ysize = size[0]
        gridx, gridy = grid_for_dumux(xsize, ysize, vs)

        h = interpolate_dumux(points, h, gridx, gridy)
        p = interpolate_dumux(points, p, gridx, gridy)
        velx = interpolate_dumux(points, velx, gridx, gridy)
        vely = interpolate_dumux(points, vely, gridx, gridy)
        velmag = interpolate_dumux(points, velmag, gridx, gridy)
        
        prefixExport_velX = f'{prefix}_v_main_'
        export2csv(prefixExport_velX, set_solid_zero(velx, filename), path, filename)

    os.chdir(script_dir)


def interpolate_dumux(points, data, gridx, gridy, method = 'linear'):
    """
    

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    gridx : TYPE
        DESCRIPTION.
    gridy : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'cubic'.

    Returns
    -------
    None.

    """
    
    return scipy.interpolate.griddata(points, data, (gridx, gridy), method=method)


def grid_for_dumux(xsize, ysize, vs):
    """
    

    Parameters
    ----------
    xsize : TYPE
        DESCRIPTION.
    ysize : TYPE
        DESCRIPTION.
    vs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    gridx, gridy = np.mgrid[0:xsize, 0:ysize]
    gridx = np.multiply(gridx, vs) 
    gridx = np.add(gridx, 0.5*vs) 
    gridy = np.multiply(gridy, vs)
    gridy = np.add(gridy, 0.5*vs) 
    
    return gridx, gridy


def export2csv(prefix, data, path, filename):
    """
    

    Parameters
    ----------
    prefix : str
        Prefix for filename.
    data : np.ndarray
        data to write.
    path : str
        path.
    filename : str
        filename.

    Returns
    -------
    None.

    """
    filename = filename.replace('.vtu', '.csv')
    np.savetxt(f'{path}/{prefix}{filename}', np.transpose(data), delimiter=',')


def set_solid_zero(field, filename):
    """
    

    Parameters
    ----------
    field : np.ndarray
        Field to be inverted.
    filename : str
        filename.

    Returns
    -------
    Inverted field.

    """
    field_corrected = field
    filename_h_map = filename.replace('-00001.vtu', '.pgm')
    h_map = cv2.imread(filename_h_map, -1)
    field_corrected[np.transpose(h_map) == 0] = 0
    
    return field_corrected