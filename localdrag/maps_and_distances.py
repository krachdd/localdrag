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

import localdrag as ld

###--------------------------------------------------------------------------------


def fully_solid_per_col(h_map01):
    """
    
    Parameters
    ----------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).

    Returns
    -------
    Dictionary of colums (column number is key) with indices of 
    voxel that are solid over entire domain height.
    
    """

    cols = {}
    
    for i in range(h_map01.shape[1]):
        cols[i] = list(np.where(h_map01[:, i] == 0)[0])
    
    return cols


def fully_solid_per_row(h_map01):
    """
    
    Parameters
    ----------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).

    Returns
    -------
    Dictionary of rows (row number is key) with indices of
    voxel that are solid over entire domain height.

    """

    rows = {}
    
    for i in range(h_map01.shape[0]):
        rows[i] = list(np.where(h_map01[i, :] == 0)[0])
    
    return rows


def get_ids_columns(geom):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.

    Returns
    -------
    Dictionary of colums (column number is key) with indices of voxel that are solid.
    
    """

    cols = {}
    
    for i in range(geom.shape[1]):
        cols[i] = list(np.where(geom[:, i] == 1)[0])
    
    return cols


def get_ids_rows(geom):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.

    Returns
    -------
    Dictionary of rows (row number is key) with indices of voxel that are solid.
    
    """
    rows = {}
    
    for i in range(geom.shape[0]):
        rows[i] = list(np.where(geom[i, :] == 1)[0])
    
    return rows


def get_solid_neighbors(index, l):
    """

    Parameters
    ----------
    index : int
        Index of current voxel.
    l     : list of int
        All indices of solid voxels in column or row.

    Returns
    -------
    ln : int 
        Index of left neighbor.
    rn : int 
        Index of right neighbor.
    
    """
    
    l.append(index)
    l.sort()
    
    if l.index(index) == 0:
        ln = l[-1]
        rn = l[1]
    # last element
    elif l.index(index) == len(l) - 1: 
        ln = l[-2]
        rn = l[0]
    else:
        ln = l[l.index(index) - 1]
        rn = l[l.index(index) + 1]
    
    l.remove(index)
    
    return int(ln), int(rn) 


def vox_distance(index, size, ln , rn):
    """

    Parameters
    ----------
    index : int
        Index of current voxel.
    size  : int
        Size of domain in direction of interest.
    ln    : int
        Index of the left neighbor.
    rn    : int
        Index of the right neighbor.

    Returns
    -------
    The "channel"-size in voxel in direction of interest;
    alias the minimum distance between ln and rn, by taking 
    periodic boundary conditions into account.
        
    """
    if ln < rn:
        return rn - ln -1
    elif ln > rn and index < rn and index < ln:
        return size - (ln - rn) -1 
    elif ln > rn and index > ln and index > rn:
        return size - (ln - rn) -1
    else:
        raise ValueError('Something went wrong with the index!')


def get_distance_two_neighbors(index, size, ln , rn):
    """

    Parameters
    ----------
    index : int
        Index of current voxel.
    size  : int
        Size of domain in direction of interest.
    ln    : int
        Index of the left neighbor.
    rn    : int
        Index of the right neighbor.   

    Returns
    -------
    The minimum distance to next solid neighbor in voxel.
        
    """
    if ln < rn:
        d1 = np.abs(index - ln)
        d2 = np.abs(rn - index)
    elif ln > rn and index < rn and index < ln:
        d1 = np.abs(rn - index)
        d2 = np.abs((size - ln)) + index
    elif ln > rn and index > ln and index > rn:
        d1 = np.abs(index - ln)
        d2 = np.abs((size -index)) + rn
    else:
        raise ValueError('Something went wrong with the index!')
    
    return min(d1, d2)
    
def get_distance_one_neighbor(index, size, n):
    """

    Parameters
    ----------
    index : int
        Index of current voxel.
    size  : int
        Size of domain in direction of interest.
    n     : int
        Index of the neighbor.

    Returns
    -------
    Minimum distance to the solid neighbor;
    takes into account, that the domain is periodic.
        
    """
    
    d1 = np.abs(n - index)
    d2 = size - np.abs(n - index)
    
    return min(d1, d2)


def get_wl_per_3d_slice(geom2d_slice, vox_per_height):
    """
    
    Parameters
    ----------
    geom2d_slice  : numpy.ndarray
        2d domain.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.
    vox_per_height : int 
        Number of voxels per h_Omega.


    Returns
    -------
    cmap : numpy.ndarray 
        Colum-wise check, w in this voxel. 
    rmap : numpy.ndarray
        Row-wise check, l  in this voxel.
    
    Compute w, l in an specific the cross section 
    for every voxel in geom2d_slice.
    Return data in integer numbers/ number of voxels per 
    cross-section. 

    """
    
    # get 2d geom 
    cols = get_ids_columns(geom2d_slice)
    rows = get_ids_rows(geom2d_slice)
    # maps for row and column wise distance
    rmap = np.zeros((geom2d_slice.shape))
    cmap = np.zeros((geom2d_slice.shape))
    
    for i in range(geom2d_slice.shape[0]):
        for j in range(geom2d_slice.shape[1]):
            # only if voxel is a fluid voxel
            if geom2d_slice[i, j] == 0:
                # current row/col add voxel id 
                if not rows[i]:
                    rmap[i, j] = 100 *vox_per_height 
                elif len(rows[i]) == 1:
                    rmap[i, j] = geom2d_slice.shape[1] - 1
                else:
                    # get left neighbor and right neighbor
                    ln, rn = get_solid_neighbors(j, rows[i])
                    # compute distance / channel size
                    rmap[i, j] = vox_distance(j, geom2d_slice.shape[1], ln, rn)
                
                if not cols[j]:
                    cmap[i, j] = 100 *vox_per_height
                elif len(cols[j]) == 1:
                    cmap[i, j] = geom2d_slice.shape[0] - 1
                else: 
                    # get left neighbor and right neighbour
                    ln, rn = get_solid_neighbors(i, cols[j])
                    # compute distance / channel size
                    cmap[i, j] = vox_distance(i, geom2d_slice.shape[0], ln, rn)
    
    return cmap, rmap


def get_wl_maps(h_map01, height, voxelsize, solidframe, channelwidth):
    """
    
    Parameters
    ----------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).
    voxelsize : float [ m ]
        Voxelsize of the domain.
    height : float [ m ]
        Height of the domain.
    channelwidth : string
        mean : return mean channel width per column
        min  : return min channel width per column
        harmonic : return harmonic mean channel width per column
    solidframe: list [bool, bool]
        If true -> there is a solid frame around 
        the domain. Important for periodic 
        mean width computation.

    Returns
    -------
    cmap : numpy.ndarray 
        Colum-wise check, mean w. 
    rmap : numpy.ndarray
        Row-wise check, mean l.
    cweight : numpy.ndarray
        weight per cross-section, colum-wise
        voxel-specific since effective cross-section
        might vary
    rweight : numpy.ndarray
        weight per cross-section, row-wise
        voxel-specific since effective cross-section
        might vary

    Compute w, l in an specific the cross section 
    for every voxel in 2d domain. It implicitly 
    reconstructs 3d information based on grey-values.
    Return data in integer numbers/ number of voxels per 
    cross-section. 

    """

    # get 2d geom 
    cols = fully_solid_per_col(h_map01)
    rows = fully_solid_per_row(h_map01)
    # maps for row and column wise distance
    rmap = np.zeros(h_map01.shape)
    cmap = np.zeros(h_map01.shape)

    # weights to account for cross-section shape
    rweight = np.zeros(h_map01.shape)
    cweight = np.zeros(h_map01.shape)

    # Sanatize h_map01
    h_map01 = h_map01_sanatize(h_map01, voxelsize, height)

    # get information regarding periodicity
    row_per = solidframe[1] 
    col_per = solidframe[0]

    vox_per_height = int(np.round(height/voxelsize)) 
    val_per_max    = 100 * vox_per_height

    row_max = h_map01.shape[0]
    col_max = h_map01.shape[1]

    # Loop for all voxels
    # could be made more efficient in the future
    for i in range(h_map01.shape[0]):
        for j in range(h_map01.shape[1]):
            # only if voxel is partialy a fluid voxel
            if h_map01[i, j] > 0:
                # Store the local height
                local_h = h_map01[i, j]

                # current row/col add voxel id 
                # non of the voxels in the row is fully solid
                # there is now solid fraction in that row whatsoever
                if (not rows[i]) and np.min(h_map01[i, :]) == 1.0:
                    rmap[i, j] = 100 * vox_per_height 
                    rweight[i, j] = 1.0
                
                # there is solid fraction in this row, but
                # no fully soilid voxel
                elif (not rows[i]) and np.min(h_map01[i, :]) < 1.0:
                    cs_as_array = h_map01[i, :]
                    index = j
                    mod_cs_as_array = mark_non_contributing_parts(np.copy(cs_as_array), index)
                    rmap[i, j] = mean_w_per_crosssection(mod_cs_as_array, vox_per_height, local_h, index, channelwidth, row_per, val_per_max, row_max)
                    rweight[i, j] = crosssection_weight(mod_cs_as_array, vox_per_height)


                # just one voxel is fully solid
                elif len(rows[i]) == 1:
                    single_index = row[i][0]
                    cs_as_array = h_map01[i, :]
                    del cs_as_array[single_index]
                    index = j
                    if single_index < index:
                        index = index -1
                    mod_cs_as_array = mark_non_contributing_parts(np.copy(cs_as_array), index)
                    # mod_cs_as_array[mod_cs_as_array == 1.] = 100 * vox_per_height
                    rmap[i, j] = mean_w_per_crosssection(mod_cs_as_array, vox_per_height, local_h, index, channelwidth, row_per, val_per_max, row_max)
                    rweight[i, j] = crosssection_weight(mod_cs_as_array, vox_per_height)
                else:
                    # get left neighbor and right neighbor
                    ln, rn = get_solid_neighbors(j, rows[i])
                    # compute distance / channel size
                    cs_as_array = get_crosssection_2neighbours(h_map01[i, :], ln, rn, j)
                    index       = get_index_2neighbours(ln, rn, j)
                    mod_cs_as_array = mark_non_contributing_parts(np.copy(cs_as_array), index)
                    rmap[i, j] = mean_w_per_crosssection(mod_cs_as_array, vox_per_height, local_h, index, channelwidth, row_per, val_per_max, row_max)
                    rweight[i, j] = crosssection_weight(mod_cs_as_array, vox_per_height)
                
                # Check specific column
                if (not cols[j]) and np.min(h_map01[:, j] == 1.0):
                    cmap[i, j] = 100 *vox_per_height
                    cweight[i, j] = 1.0

                elif (not cols[j]) and np.min(h_map01[:, j] < 1.0):
                    cs_as_array = h_map01[:, j]
                    index = i
                    mod_cs_as_array = mark_non_contributing_parts(np.copy(cs_as_array), index)
                    cmap[i, j] = mean_w_per_crosssection(mod_cs_as_array, vox_per_height, local_h, index, channelwidth, col_per, val_per_max, col_max)
                    cweight[i, j] = crosssection_weight(mod_cs_as_array, vox_per_height)

                elif len(cols[j]) == 1:
                    single_index = cols[j][0]
                    cs_as_array = h_map01[:, j]
                    del cs_as_array[single_index]
                    index = i
                    if single_index < index:
                        index = index -1
                    mod_cs_as_array = mark_non_contributing_parts(np.copy(cs_as_array), index)
                    cmap[i, j] = mean_w_per_crosssection(mod_cs_as_array, vox_per_height, local_h, index, channelwidth, col_per, val_per_max, col_max)
                    cweight[i, j] = crosssection_weight(mod_cs_as_array, vox_per_height)

                else: 
                    # get left neighbor and right neighbor
                    ln, rn = get_solid_neighbors(i, cols[j])
                    cs_as_array = get_crosssection_2neighbours(h_map01[:, j], ln, rn, i)
                    index       = get_index_2neighbours(ln, rn, i)
                    mod_cs_as_array = mark_non_contributing_parts(np.copy(cs_as_array), index)
                    cmap[i, j] = mean_w_per_crosssection(mod_cs_as_array, vox_per_height, local_h, index, channelwidth, col_per, val_per_max, col_max)
                    cweight[i, j] = crosssection_weight(mod_cs_as_array, vox_per_height)

    return cmap, rmap, cweight, rweight


def get_wl_maps_3d(geom):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.

    Returns
    -------
    w_3d : numpy.ndarray 
        3d information w(x) in number of voxels. 
    l_3d : numpy.ndarray
        3d information l(x) in number of voxels.

    """
    
    # Get a eqivalent domains to store ratios
    w_3d = np.zeros((geom.shape)) 
    l_3d = np.zeros((geom.shape)) 

    for i in range(geom.shape[0]):
        w_3d[i, :, :], l_3d[i, :, :] = get_wl_per_3d_slice(geom[i, :, :], geom.shape[0])

    return w_3d, l_3d


def get_wl_per_stack_3d(w_3d, l_3d, channelwidth):
    """
    
    Parameters
    ----------
    w_3d : numpy.ndarray 
        3d information w(x) in number of voxels. 
    l_3d : numpy.ndarray
        3d information l(x) in number of voxels.
    MINW : bool
        If true -> use min channel width
        else -> use mean channel width

    Returns
    -------
    mean_w_3d : numpy.ndarray 
        3d information w(x) in number of voxels. 
    mean_l_3d : numpy.ndarray
        3d information l(x) in number of voxels.
    cweight : numpy.ndarray
        weight per cross-section, colum-wise
        voxel-specific since effective cross-section
        might vary
    rweight : numpy.ndarray
        weight per cross-section, row-wise
        voxel-specific since effective cross-section
        might vary

    """
    
    mean_w_map = np.zeros((w_3d.shape[1], w_3d.shape[2]))
    mean_l_map = np.zeros((w_3d.shape[1], w_3d.shape[2]))
    cweight    = np.zeros((w_3d.shape[1], w_3d.shape[2]))
    rweight    = np.zeros((w_3d.shape[1], w_3d.shape[2]))

    for j in range(w_3d.shape[1]):
        for k in range(w_3d.shape[2]):
            w_temp = w_3d[:, j, k]
            l_temp = l_3d[:, j, k]
            # Delete all zeros from the array
            w_temp = w_temp[w_temp != 0]
            l_temp = l_temp[l_temp != 0]

            # Avoid empty array warings
            if w_temp.size == 0:
                mean_w_map[j, k] = 0.0
            else:
                if channelwidth == 'min':
                    mean_w_map[j, k] = np.min(w_temp)
                elif channelwidth == 'mean':
                    mean_w_map[j, k] = np.mean(w_temp)
                elif channelwidth == 'harmonic':
                    mean_w_map[j, k] = scipy.stats.hmean(w_temp)
                else:
                    raise ValueError('No valid method defined for channelwidth!')
                cweight = np.max(w_temp)


            if l_temp.size == 0:
                mean_l_map[j, k] = 0.0
            else:
                if channelwidth == 'min':
                    mean_l_map[j, k] = np.min(l_temp)
                elif channelwidth == 'mean':
                    mean_l_map[j, k] = np.mean(l_temp)
                elif channelwidth == 'harmonic':
                    mean_l_map[j, k] = scipy.stats.hmean(l_temp)
                else:
                    raise ValueError('No valid method defined for channelwidth!')
                rweight = np.max(l_temp)

    return mean_w_map, mean_l_map, cweight, rweight



def get_hmap(geom, voxelsize, vox_per_height):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.
    voxelsize : float [ m ]
        Voxelsize of the domain.
    vox_per_height : int 
        Number of voxels per h_Omega.


    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    
    """

    h_map01 = ld.evaluate3d.get_hmap01(geom)
    
    return np.multiply(vox_per_height * voxelsize, h_map01)


def scale_hmap(h_map01, height):
    """
    
    Parameters
    ----------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).
    height : float
        h_Omega of the domain  [ m ]

    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    
    """

    return np.multiply(height, h_map01)

def unscale_hmap(h_map_scaled, height):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    height : float
        h_Omega of the domain  [ m ]

    Returns
    -------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).
    
    """

    return np.divide(h_map_scaled, height, out=np.zeros_like(h_map_scaled), where=height!=0)


def h_map_settle_rounding_error(h_map_scaled, voxelsize):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    voxelsize : float [ m ]
        Voxelsize of the domain.

    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize

    Remove rounding errors introduced by the data format.
    
    """

    h_map_scaled = h_map_scaled/voxelsize
    h_map_scaled = np.round(h_map_scaled)
    h_map_scaled = h_map_scaled*voxelsize

    return h_map_scaled


def h_map01_sanatize(h_map01, voxelsize, height):

    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    voxelsize : float [ m ]
        Voxelsize of the domain.
    height : float [ m ]
        Height of the domain.
    vox_per_height : int 
        Number of voxels per h_Omega.

    Returns
    -------
    h_map01 : numpy ndarray
        2d, relative height of geometry. Scaled
        between 0 (fully blocked/solid voxel) and 
        1 (fully fluid voxel).

    Remove rounding errors introduced by the data format.
    
    """

    # names not matching here!
    h_map01 = scale_hmap(h_map01, height)
    h_map01 = h_map_settle_rounding_error(h_map01, voxelsize)
    h_map01 = unscale_hmap(h_map01, height)

    return h_map01


def get_hratio_map(h_map_scaled, voxelsize, height):
    
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    voxelsize : float [ m ]
        Voxelsize of the domain.
    height : float [ m ]
        Height of the domain.


    Returns
    -------
    h_Omega_by_hx : numpy.ndarray 
        2d array/map contaning relative heigth of the channel. 
    
    """

    h_Omega_map = np.multiply(np.ones(h_map_scaled.shape), height) 

    return np.divide(h_Omega_map, h_map_scaled, out=np.zeros_like(h_Omega_map), where=h_map_scaled!=0)


def apply_empirical_wh_relation(ratios):
    """
    Parameters
    ----------
    ratios : numpy.ndarray
        w/l to h ratio.
    
    Returns
    -------
    f_map : numpy.ndarray
        2d map containing factors to use in DUMUX simulator. 
    
    """
    
    #factor_map_sanity_check() TODO
    f_map = np.zeros((ratios.shape), dtype = np.float64)
    
    for j in range(ratios.shape[0]):
        for k in range(ratios.shape[1]):
            f_map[j, k] = ld.empirical_functions.lambda_wh(ratios[j, k])

    return f_map


def apply_empirical_grad_relation(gradient):
    """
    Parameters
    ----------
    gradient : numpy.ndarray
        central difference quotient of the h-values of the domain
    
    Returns
    -------
    f_map : numpy.ndarray
        2d map containing factors to use in DUMUX simulator. 
    """
    
    #factor_map_sanity_check() TODO
    f_map = np.zeros((gradient.shape), dtype = np.float64)
    
    for j in range(gradient.shape[0]):
        for k in range(gradient.shape[1]):
            f_map[j, k] = ld.empirical_functions.lambda_gi(gradient[j, k])

    return f_map


def label_successive_zeros(h_map_scaled):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    
    Returns
    -------
    label: array assingning each voxel to a cross-section
    
    """
    
    label = np.zeros_like(h_map_scaled)
    current_label = 2
    NO_FLUID_INDICATOR = 0

    for i in range(h_map_scaled.shape[0]):
        same_label = True
        for j in range(h_map_scaled.shape[1]):
            if h_map_scaled[i, j] == NO_FLUID_INDICATOR:
                current_label += 1
                label[i, j] = NO_FLUID_INDICATOR
                same_label = False
            else:
                label[i, j] = current_label
        if not same_label:
            current_label += 1
    
    return label


def label_successive_zeros_periodic(h_map_scaled):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    
    Returns
    -------
    label: array assingning each voxel to a cross-section
    
    """

    label = np.zeros_like(h_map_scaled)
    current_label = 2
    NO_FLUID_INDICATOR = 0

    for i in range(h_map_scaled.shape[0]):
        same_label = True
        for j in range(h_map_scaled.shape[1]):
            if h_map_scaled[i, j] == NO_FLUID_INDICATOR:
                current_label += 1
                label[i, j] = NO_FLUID_INDICATOR
                same_label = False
            else:
                label[i, j] = current_label
        if not same_label:
            current_label += 1
        if label[i, 0] != NO_FLUID_INDICATOR and label[i, -1] != NO_FLUID_INDICATOR:
            label[i, label[i] == label[i, -1]] = label[i, 0]
        current_label += 1
    
    return label

def label_successive_zeros_columnwise(h_map_scaled):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    
    Returns
    -------
    label: array assingning each voxel to a cross-section
    
    """
    
    labeled_transposed = label_successive_zeros(h_map_scaled.T)  # Apply row-wise labeling to transposed h_map_scaled
    return labeled_transposed.T


def label_successive_zeros_columnwise_periodic(h_map_scaled):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    
    Returns
    -------
    label: array assingning each voxel to a cross-section
    
    """
    
    labeled_transposed = label_successive_zeros_periodic(h_map_scaled.T)  # Apply row-wise labeling to transposed h_map_scaled
    
    return labeled_transposed.T

def label_2d_geom(h_map_scaled, solidframe):
    """
    
    Parameters
    ----------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize

    solidframe: list [bool, bool]
        If true -> there is a solid frame around 
        the domain. Important for periodic 
        mean width computation.
    
    Returns
    -------
    label: array assingning each voxel to a cross-section
    
    """

    if solidframe[0] == True:
        rowwise_label = label_successive_zeros(h_map_scaled)
    else:
        rowwise_label = label_successive_zeros_periodic(h_map_scaled)
    
    if solidframe[1] == True:
        colwise_label = label_successive_zeros_columnwise(h_map_scaled)
    else:
        colwise_label = label_successive_zeros_columnwise_periodic(h_map_scaled)

    return colwise_label, rowwise_label

def assign_values(labels, values, crosssection):
    """

    Parameters
    ----------
    labels: numpy.ndarray
        each voxel which is part of a cross-section has same label
    values: 
        values, voxel specific 
    crosssection : string
        mean : get a mean width/length/height per crosssection
        min  : get the min w/l/h per crosssection
        max  : only for csweight
        different : stay with one w/h or l/ ratio per column;
                    this means different w/h ratios per 
                    crosssection 

    Returns
    -------
    mean_vals: assign min value per label to all voxels with same label

    """

    mean_vals = np.zeros_like(labels)
    
    for label in np.unique(labels):
        if label != 0 and label != 1:  # Skip for zeros and ones
            if crosssection == 'mean':
                mean_vals[labels == label] = np.mean(values[labels == label])
            elif crosssection == 'min':
                mean_vals[labels == label] = np.min(values[labels == label])
            elif crosssection == 'max':
                mean_vals[labels == label] = np.max(values[labels == label])
            else:
                raise ValueError('No valid method defined for crosssection!')
    return mean_vals


def mean_w_per_crosssection(array, vox_per_height, local_h, index, channelwidth, periodic, val_max, check_num):
    """

    Parameters
    ----------
    array : numpy.ndarray 1D 
        containing cross-section specific values
    vox_per_height : int 
        Number of voxels per h_Omega.
    local_h : float
        relative height at current voxel
    index : int
        index of current voxel in array
    channelwidth : string
        mean : return mean channel width per column
        min  : return min channel width per column
        harmonic : return harmonic mean channel width per column
    periodic : bool
        if domain is periodic in current direction
    val_max: float
        assign this if check_num matches and 
        periodic is true.
    check_num : int
        max_value to check for if all is fluid so 
        periodicity plays a role.


    Think of this algorithm like this:

    It takes a 1D array with grey values of the height
    of one cross-section in 2D
    
    a = [0.1 0.6 0.6 0.8 0.9 0.9 0.7 0.2]
    
    and implicitly reconstructs to 

    b = 
    [1, 1, 1, 1, 1, 1, 1, 1]
    [0, 1, 1, 1, 1, 1, 1, 1]
    [0, 1, 1, 1, 1, 1, 1, 0]
    [0, 1, 1, 1, 1, 1, 1, 0]
    [0, 1, 1, 1, 1, 1, 1, 0]
    [0, 1, 1, 1, 1, 1, 1, 0]
    [0, 0, 0, 1, 1, 1, 1, 0]
    [0, 0, 0, 1, 1, 1, 0, 0]
    [0, 0, 0, 0, 1, 1, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0]

    0 is precipitate, and since this is one cross section 
    the bigger picture would look like

    b_bigger = 
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    using this one can compute the mean width and height 
    by counting ones and relating it to the respective shape


    Resulting in Mean number of ones per row:
    Row 1: 1.0
    Row 2: 0.875
    Row 3: 0.75
    Row 4: 0.75
    Row 5: 0.75
    Row 6: 0.75
    Row 7: 0.5
    Row 8: 0.375
    Row 9: 0.25
    Row 10: 0.0

    Returns
    -------
    specific w/l for one voxel dependend on cross-section properties

    """

    num_columns = array.shape[0]
    num_rows = vox_per_height
    # Initialize a list to store the total count
    total_fluid_per_row = [0] * num_rows
    array_scaled = np.multiply(array, vox_per_height)

    for i in range(array_scaled.shape[0]):
        for j in range(int(min(array_scaled[i], vox_per_height))):
            total_fluid_per_row[j] += 1

    total_fluid_per_row = np.asarray(total_fluid_per_row)

    if periodic == True:
        total_fluid_per_row[total_fluid_per_row == check_num] = val_max

    num_local_rows = int(round(vox_per_height *  local_h))
    local_rows = total_fluid_per_row[0:num_local_rows]

    if local_rows.size == 0:
        print(f"Return 0: {local_h}")
        return 0.0

    else:
        if channelwidth == 'min':
            return np.min(local_rows)
        elif channelwidth == 'mean':
            return np.mean(local_rows)
        elif channelwidth == 'harmonic':
            return scipy.stats.hmean(local_rows)
        else:
            raise ValueError('No valid method defined for channelwidth!')


def crosssection_weight(array, vox_per_height):
    """

    Parameters
    ----------
    array : numpy.ndarray 1D 
        containing cross-section specific values
    vox_per_height : int 
        Number of voxels per h_Omega.

    Returns
    -------
    specific weight per voxel, is 1 for perfectly rect 
    channels and <1 for other arbitrary cross-sections

    """

    num_columns = array.shape[0]
    num_rows = vox_per_height * np.max(array)

    # Maximal area per cross-section in voxels
    area_max = num_columns * num_rows

    # actual area, counts all fluid part
    discrete_h = np.multiply(array, vox_per_height)
    area_cs = np.sum(discrete_h)

    return area_cs/area_max



def mark_non_contributing_parts(array, index):
    """

    Parameters
    ----------
    array : numpy.ndarray 1D 
        containing cross-section specific values
    index : int
        index of current voxel in array

    Returns
    -------
    array : numpy.ndarray 1D 
    
    """
    
    min_positive = array[index]
    # Sweep in positive direction
    for ii in range(index + 1, array.shape[0], 1):
        if array[ii] < min_positive:
            min_positive = array[ii]
        else:
            array[ii] = min_positive

    min_negative = array[index]
    # Sweep in negative direction
    for i in range(index - 1, -1, -1):
        if array[i] < min_negative:
            min_negative = array[i]
        else:
            array[i] = min_negative

    return array


def get_crosssection_2neighbours(a, ln, rn, index):
    """

    Parameters
    ----------

    size  : int
        Size of domain in direction of interest.
    ln    : int
        Index of the left neighbor.
    rn    : int
        Index of the right neighbor.
    index : int
        Index of current voxel.

    Returns
    -------
    The cross-section in direction of interest;
    between ln and rn, by taking 
    periodic boundary conditions into account.
        
    """
    if ln < rn:
        if index < ln or index > rn:
            raise ValueError("Index not possible")
        return a[ln+1:rn]
    elif ln > rn and index < rn and index < ln:
        return np.concatenate((a[0:rn], a[ln+1::]), axis = 0)
    elif ln > rn and index > ln and index > rn:
        return np.concatenate((a[0:rn], a[ln+1::]), axis = 0)


def get_index_2neighbours(ln, rn, index):
    """

    Parameters
    ----------

    size  : int
        Size of domain in direction of interest.
    ln    : int
        Index of the left neighbor.
    rn    : int
        Index of the right neighbor.
    index : int
        Index of current voxel.

    Returns
    -------
    index of current voxel in array
        
    """
    if ln < rn:
        if index < ln or index > rn:
            raise ValueError("Index not possible")
        return index - ln - 1
    elif ln > rn and index < rn and index < ln:
        return index
    elif ln > rn and index > ln and index > rn:
        return rn + index -ln -1
