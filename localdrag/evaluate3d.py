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
import fluidfoam

import localdrag as ld

###--------------------------------------------------------------------------------


def collapse(array):
    """ 

    Parameters
    ----------
    array : np.ndarray
        Collapse arrays first axis by assigning mean value
        
    
    Returns
    -------
    2d array with shape [array.shape[1], array.shape[2]]
    
    """

    a_2d = np.zeros((array.shape[1], array.shape[2]), dtype = np.float64)
    for j in range(array.shape[1]):
        for k in range(array.shape[2]): 
            a_2d[j, k] = np.mean(array[:, j, k])
            
    return a_2d


def geom_informed_collapse(geom, array):
    """ 
    
    Parameters
    ----------
    geom : np.ndarray, bool 
        : geometry
    array : np.ndarray
        Collapse arrays first axis by averagin value if geometry is fluid.
        
    
    Returns
    -------
    2d array with shape [array.shape[1], array.shape[2]]
    
    """

    if geom.shape != array.shape:
        raise ValueError(f'Geometry and array must have same shape!')
    
    a_2d = np.zeros((array.shape[1], array.shape[2]), dtype = np.float64)    
    for j in range(array.shape[1]):
        for k in range(array.shape[2]):
            # init counter and sum of scalars  
            c = 0
            sum_a = 0
            for i in range(array.shape[0]):
                if geom[i, j, k] == 0:
                    c += 1
                    sum_a += array[i, j, k]
            if c == 0:
                a_2d[j, k] = 0
            else:
                a_2d[j, k] = np.divide(sum_a, c)
            
    return a_2d


def get_3d_fields(path, fn, size, voxelsize, frames = []):
    """
    
    Parameters
    ----------
    path : string
        Path in storage to save file in. 
    fn   : string
        Filename of the input/geometry file.
    size : [int, int, int]
        Size of the domain.
    voxelsize : float [ m ]
        Voxelsize of the domain.
    frames : list of ints
        List of directions in which frames should be removed.

    Returns
    -------
    Dictionary with all 3d-fields.
    
    Reads standart input from Stokes Solver simulations with fixed filenames.

    """

    fields = {}

    # read all fields
    for p in ld.constants.STOKES_FILE_PREFIXES:
        # Import data
        if p == 'geom':
            print(f'get_3d_fields:\t Read File {fn}.')
            fields[p] = ld.wrap_import.read_3d_rawfile(fn, size, voxelsize)
        else:
            if not os.path.isfile(f'{p}_{fn}'):
                print(f'get_3d_fields:\t File {p}_{fn} does not exist!')
            else: 
                print(f'get_3d_fields:\t Read File {p}_{fn}.')
                fields[p] = ld.wrap_import.read_3d_rawfile(f'{p}_{fn}', size, voxelsize, dtype = np.float64)

        # Remove all frames for all fields
        for i in frames:
            if p in fields.keys():
                fields[p], size_ = ld.wrap_import.remove_frame(fields[p], axis = i)
    
    return fields


def get_2d_averaged_fields(path, fn,  size, voxelsize, frames = []):
    """
    
    Parameters
    ----------
    path : string
        Path in storage to save file in. 
    fn   : string
        Filename of the input/geometry file.
    size : [int, int, int]
        Size of the domain.
    voxelsize : float [ m ]
        Voxelsize of the domain.
    frames : list of ints
        List of directions in which frames should be removed.

    Returns
    -------
    Dictionary with all 3d-fields.
    
    Reads standart input from Stokes Solver simulations with fixed filenames.

    """

    # remove .raw form fn 
    filename = fn.replace('.raw', '')

    fields2d = {}
    
    # read all fields
    fields = get_3d_fields(path, fn, size, voxelsize, frames)

    if sorted(fields.keys()) == sorted(ld.constants.STOKES_FILE_PREFIXES):
        prefixs = ld.constants.STOKES_FILE_PREFIXES
        print(f'All files ({ld.constants.STOKES_FILE_PREFIXES}) available.')
    else:
        prefixs = fields.keys()
        print(f'Following files ({fields.keys()}) available.')

    for p in prefixs:
        # collapse depends on prefix type
        if p == 'geom':
            fields2d[p] = collapse(fields[p])
        else:
            fields2d[p] = geom_informed_collapse(fields['geom'], fields[p])

    return fields2d

def get_hmap01(geom):
    """
    
    Parameters
    ----------
    geom  : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.

    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize

    If a value in the resulting map is zero -> channel at
    this point is completly blocked by solid
    
    """
    
    # Create a 2d array tp store output
    h_map = np.zeros((geom.shape[1], geom.shape[2]), dtype = np.float64) 
    # loop over all columns and rows of the geometry
    for j in range(geom.shape[1]):
        for k in range(geom.shape[2]):
            # compute ratio of fluid to number of voxels in domain height  
            h_map[j,k] = 1.0 - np.mean(geom[:, j, k])
            
    return h_map



def read_openfoam_results(path, filename, timestep):
    """

    """

    # get the absolute path of the simulation case
    fullpath = os.path.join(path, filename) 

    vel      = fluidfoam.readvector(fullpath, timestep, 'U')
    x, y, z  = fluidfoam.readmesh(fullpath)
    press    = fluidfoam.readscalar(fullpath, timestep, 'p')

    return x, y, z, vel, press


def collapse_openfoam_results(path, filename, timestep, size, voxelsize)
    """
    """

    x, y, z, vel, press = read_openfoam_results(path, filename, timestep)

    # get domain sizes in meter
    sim_dom_size      = np.asarray(size) * voxelsize
    total_mesh_number = size[0] * size[1] * size[2]

    # velocity/pressure data structured
    vel_structured   = np.zeros((3, total_mesh_number), dtype = float)
    press_structured = np.zeros((1, total_mesh_number), dtype = float)

    # step 2: unstructural mesh to structural mesh
    for k in np.arange(size[2]):
        for j in np.arange(size[1]):
            for i in np.arange(size[0]):
                condition = (x >= i*voxelsize) & (x < (i+1)*voxelsize) & (y >= j*voxelsize) & (y < (j+1)*voxelsize) & (z >= k*voxelsize) & (z < (k+1)*voxelsize)
                vel_structured[0,(i+j*size[0]+k*size[1]*size[0])] = np.mean(vel[0][condition]) # global index: i+j*size[0]+k*size[1]
                vel_structured[1,(i+j*size[0]+k*size[1]*size[0])] = np.mean(vel[1][condition])
                vel_structured[2,(i+j*size[0]+k*size[1]*size[0])] = np.mean(vel[2][condition])
                press_structured[(i+j*size[0]+k*size[1]*size[0])] = np.mean(press[condition])

    # Average velocity/pressure
    vel_structured_x = vel_structured[0].reshape((size[2], (size[0]*size[1]))) # every row represents the velocity of every x-y plan; Uusx is the X-velocity of the structured mesh
    vel_structured_y = vel_structured[1].reshape((size[2], (size[0]*size[1])))
    vel_structured_z = vel_structured[2].reshape((size[2], (size[0]*size[1])))
    press_structured = press_structured.reshape((size[2], (size[0]*size[1])))

    vel_structured_x = np.nan_to_num(vel_structured_x)
    vel_structured_y = np.nan_to_num(vel_structured_y)
    vel_structured_z = np.nan_to_num(vel_structured_z)


    # averate the results in z directions
    vel_structured_x = np.mean(vel_structured_x, axis=0).reshape(size[0], size[1])
    vel_structured_y = np.mean(vel_structured_y, axis=0).reshape(size[0], size[1])
    vel_structured_z = np.mean(vel_structured_z, axis=0).reshape(size[0], size[1])
    press_structured = np.nanmean(press_structured, axis=0).reshape(size[0], size[1])

    ld.write_maps.write2txt(path, filename, 'velx', vel_structured_x)
    ld.write_maps.write2txt(path, filename, 'vely', vel_structured_y)
    ld.write_maps.write2txt(path, filename, 'velz', vel_structured_z)
    ld.write_maps.write2txt(path, filename, 'press', press_structured)

