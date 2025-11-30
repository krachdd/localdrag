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
import os, cv2, vtk
from PIL import Image
from pyevtk.hl import pointsToVTK as ptovtk
from vtk.util import numpy_support as VN

###--------------------------------------------------------------------------------

def sanity_check(filename, size, vsize):
    """
    
    Parameters
    ----------
    filename : string
        Path to file (.raw).
    size : [int, int, int]
        Size of the domain. 
    voxelsize : float [ m ]
        Voxelsize of the domain.

    Returns
    -------
    Nothing.
    
    """
    
    if not os.path.isfile(filename):
        raise ValueError(f'File {filename} does not exist!')
    
    if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
        raise ValueError('Work with Positive Numbers for VoxelCount Only')
        
    if vsize <= 0:
        raise ValueError('Work with Positive Numbers for VoxelSize Only')
    
    


def read_3d_rawfile(filename, size, voxelsize, reshaped = True, dtype = np.uint8, order = 'F'):
    """
    
    Parameters
    ----------
    filename : string
        Path to file (.raw).
    size : [int, int, int]
        Size of the domain. 
    voxelsize : float
        Voxelsize of the domain.
    reshaped : bool, optional
        Return the domain array reshaped with to size or not. 
        The default is True.
    dtype : numpy datatype, optional
        Datatype of input. 
        The default is np.uint8.
        Use np.float64 for pressure and velocity files from 
        Stokes Solver simulations.
    order : string, optional
        Byte order of file in storage. 
        The default is 'F'.

    Returns
    -------
    array : numpy.ndarray
        Raw geometry.
    
    """
    
    sanity_check(filename, size, voxelsize)
    xsize, ysize, zsize = int(size[0]), int(size[1]), int(size[2])
    voxelsize = float(voxelsize)
    
    count = xsize * ysize * zsize
    
    rawfile = open(filename,'rb')
    
    array = np.fromfile(rawfile, dtype=dtype, count=count)
    
    if reshaped:
        array = np.reshape(array, (xsize, ysize, zsize), order = order)
    else:
        print(f'Data is not reshaped to 3D, size: {array.shape}')

    return array


def remove_frame(array, axis = 0):
    """
    
    Parameters
    ----------
    filename : numpy.ndarray
        Raw geometry.
    axis : int
        Direction in which frame should be removed.

    Returns
    -------
    array : numpy.ndarray
        Raw geometry without solid frame.
    array.shape : list [int, int, int]
        Size of array after removing frame. 
    
    """
    if array.ndim == 2:
        if axis == 0:
            array = array[2:array.shape[0]-2, :] 
        if axis ==  1:
            array = array[:, 2:array.shape[1]-2]

    elif array.ndim == 3:
        if axis == 0:
            array = array[2:array.shape[0]-2, :, :] 
        if axis == 1:
            array = array[:, 2:array.shape[1]-2, :]
        if axis == 2:
            array = array[:, :, 2:array.shape[0]-2]

    else:
        raise ValueError(f'Dimension of input array {array.ndim} incorrect.')
            
    return array, array.shape

def getNumVoxelFromName(fn):
    """

    Parameters
    ----------
    filename : string

    Returns
    -------
    size : list [int, int, int]
        Size from file name. 
    
    """
    
    fn1 = fn.split("domain_")
    fn2 = fn1[1].split("_vs")
    fn3 = fn2[0].split("_")
    
    size = [int(fn3[0]), int(fn3[1]), int(fn3[2])]
    
    return size


def gethvoxFromName(fn):
    """

    Parameters
    ----------
    filename : string

    Returns
    -------
    voxelsize : hvox
    
    """

    fn1 = fn.split("_hvox")
    fn2 = fn1[1].split("_")
    
    return int(fn2[0])


def getVoxelSizeFromName(fn):
    """

    Parameters
    ----------
    filename : string

    Returns
    -------
    voxelsize : float [ m ]
        Voxelsize of the domain.
    
    """

    fn1 = fn.split("_vs")
    fn2 = fn1[1].split("_")
    
    return float(fn2[0])


def read_pgm(filename):
    """
    
    Parameters
    ----------
    filename : *.pgm file
        h_map data in 2D

    Returns
    -------
    array : numpy.ndarray
        2d-Raw geometry without solid frame.
    array.shape : list [int, int]
        Size of array after removing frame. 
    
    """
    # array = cv2.imread(filename, cv2.IMREAD_UNCHANGED) ??
    array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    array = array/255 

    return array, array.shape


def h_map_frame(h_map01):
    """
    
    Parameters
    ----------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).

    Returns
    -------
    h_map01 with frame.

    """
    frame = np.zeros((1, h_map01.shape[1]))
    array = np.concatenate((frame, h_map01, frame), axis = 0)

    return array


def read_tif(filename, scaled = True):
    """
    
    """
    im = Image.open(filename)
    array = np.array(im)
    if scaled:
        array = array/255

    return array 


def getNumVoxelFrom2DName(fn):
    """

    Parameters
    ----------
    filename : string

    Returns
    -------
    size : list [int, int, int]
        Size from file name. 
    
    """
    
    fn1 = fn.split('shape_')
    fn2 = fn1[1].split('_vs')
    fn3 = fn2[0].split('_')
    
    size = [int(fn3[0]), int(fn3[1])]
    
    return size



def openvtu_2d_dumux(filename, options_list):
    """
    
    Parameters
    ----------
    filename : str
        DESCRIPTION.
    options_list : list
        containing all fields that should be loaded
    Returns
    -------
    Dictionary: {key(entry in options_list): field as np.ndarray}
    
    
    """
    # MISSING SANITY CHECK!! 

    dict_to_return = {}
    
    print(f'Read vtk-file: {filename}')
    # Initalize Reader and get data
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    
    # Read the Particle Positions
    vtk_points = data.GetPoints().GetData()
    point_position = VN.vtk_to_numpy(vtk_points)
    coordinates = np.array(vtk_points)
    converter = vtk.vtkCellDataToPointData()  
    converter.SetInputConnection(reader.GetOutputPort() )
    converter.Update() 

    print(f'Read Positions: Number of loaded points {point_position.shape[0]}')

    # Add Points and TypeId to return dictionary
    dict_to_return['Points'] = point_position


    reader.GetOutputPort() 
    # Loop over all fields to extract and save them in return dict
    for field in options_list:
        # vtk_field = data.GetCellData().GetArray(field)

        point_field = np.array(converter.GetOutput().GetPointData().GetArray(field))

        # point_field = VN.vtk_to_numpy(vtk_field)
        print(f'Load field {field}')
        dict_to_return[field] = point_field
    

    return dict_to_return


def extract_fields(data):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    points   = data['Points'][:,0:2]
    velmag   = np.sqrt(data['velocity_liq (m/s)'][:, 0]**2 
               + data['velocity_liq (m/s)'][:, 1]**2 
               + data['velocity_liq (m/s)'][:, 2]**2)
    h        = data['relheight']
    p        = data['p']
    velx     = data['velocity_liq (m/s)'][:,0]
    vely     = data['velocity_liq (m/s)'][:,1]
    
    velmag[np.isnan(velmag)] = 0
    velx[np.isnan(velx)] = 0
    vely[np.isnan(vely)] = 0

    return points, velx, vely, velmag, p , h




