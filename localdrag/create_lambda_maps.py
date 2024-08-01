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

import localdrag as ld

import matplotlib.pyplot as plt

###--------------------------------------------------------------------------------

def lambda_wh_map(h_map01, voxelsize, height, channelwidth, solidframe, crosssection, cs_weight):
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
    crosssection : string
        mean : get a mean width/length/height per crosssection
        min  : get the min w/l/h per crosssection
        different : stay with one w/h or l/ ratio per column;
                    this means different w/h ratios per 
                    crosssection 
    cs_weight : bool 
        weight the w/h ratio by the relative area. This is introduced
        since we approximate arbitrary cross-sections by rectangles. 
        For higher perimeter-to-area ratios this results in an error which 
        is reduces by this weight. 

    Returns
    -------
    lambda1 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in main direction (z, 2nd dir of 2d array, main pressure gradient)
    lambda2 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in perpendicular direction 
    
    Factor maps here are only based on 'channel' ratios.
    
    """
    # h_Omega
    h_map_scaled = ld.maps_and_distances.scale_hmap(h_map01, height)
    h_map_scaled = ld.maps_and_distances.h_map_settle_rounding_error(h_map_scaled, voxelsize)
    
    # Get labels per cross-section
    col_labels, row_labels = ld.maps_and_distances.label_2d_geom(h_map_scaled, solidframe) 

    # Results in same ratios as if it would in real dimensions
    w_map, l_map, w_weight, l_weight = ld.maps_and_distances.get_wl_maps(h_map01, height, voxelsize, solidframe, channelwidth)

    vox_per_height = int(np.round(height/voxelsize))

    # Sanatize h_map01
    h_map01 = ld.maps_and_distances.h_map01_sanatize(h_map01, voxelsize, height)

    if crosssection == 'mean':
        # Assign values per cross-section
        mean_w_map = ld.maps_and_distances.assign_values(col_labels, w_map, crosssection)
        mean_l_map = ld.maps_and_distances.assign_values(row_labels, l_map, crosssection)
        mean_h_map_col = ld.maps_and_distances.assign_values(col_labels, h_map01 * vox_per_height, crosssection)
        mean_h_map_row = ld.maps_and_distances.assign_values(row_labels, h_map01 * vox_per_height, crosssection)
        mean_w_weight = ld.maps_and_distances.assign_values(col_labels, w_weight, crosssection)
        mean_l_weight = ld.maps_and_distances.assign_values(row_labels, l_weight, crosssection)
    elif crosssection == 'min':
        # Assign values per cross-section
        mean_w_map = ld.maps_and_distances.assign_values(col_labels, w_map, crosssection)
        mean_l_map = ld.maps_and_distances.assign_values(row_labels, l_map, crosssection)
        mean_h_map_col = ld.maps_and_distances.assign_values(col_labels, h_map01 * vox_per_height, 'mean')
        mean_h_map_row = ld.maps_and_distances.assign_values(row_labels, h_map01 * vox_per_height, 'mean')
        mean_w_weight = ld.maps_and_distances.assign_values(col_labels, w_weight, 'mean')
        mean_l_weight = ld.maps_and_distances.assign_values(row_labels, l_weight, 'mean')
    elif crosssection == 'different':
        mean_w_map = w_map
        mean_l_map = l_map
        mean_h_map_col = h_map01* vox_per_height
        mean_h_map_row = h_map01* vox_per_height
        mean_w_weight = w_weight
        mean_l_weight = l_weight
    else:
        raise ValueError('No valid method defined for crosssection!')

    # print(f'Assigned mean values: shape: {mean_w_map[1:-1].shape}, w: {np.mean(mean_w_map[1:-1])}, l: {np.mean(mean_l_map[1:-1])}')

    # Compute ratios
    wh_ratio = np.divide(mean_w_map, mean_h_map_col, out=np.zeros_like(mean_w_map), where=mean_h_map_col!=0)
    lh_ratio = np.divide(mean_l_map, mean_h_map_row, out=np.zeros_like(mean_l_map), where=mean_h_map_row!=0)

    # print(f'wh_ratios: shape: {wh_ratio[1:-1].shape}, w: {np.mean(wh_ratio[1:-1])}, l: {np.mean(lh_ratio[1:-1])}')
    if cs_weight == True:
        wh_ratio = np.multiply(wh_ratio, mean_w_weight)
        lh_ratio = np.multiply(lh_ratio, mean_l_weight)
    # print(f'weighted wh_ratios: shape: {wh_ratio[1:-1].shape}, w: {np.mean(wh_ratio[1:-1])}, l: {np.mean(lh_ratio[1:-1])}')


    # Actual factor computation
    lambda1 = ld.maps_and_distances.apply_empirical_wh_relation(wh_ratio)
    lambda2 = ld.maps_and_distances.apply_empirical_wh_relation(lh_ratio)
    
    # Normalize factor maps
    lambda1 = np.divide(lambda1, ld.constants.DUMUX_PREFACTOR)
    lambda2 = np.divide(lambda2, ld.constants.DUMUX_PREFACTOR)

    return lambda1, lambda2


def lambda_wh_map_3d(geom, voxelsize, channelwidth, solidframe, crosssection, cs_weight):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.
    voxelsize : float [ m ]
        Voxelsize of the domain.
    channelwidth : string
        mean : return mean channel width per column
        min  : return min channel width per column
        harmonic : return harmonic mean channel width per column
    solidframe: list [bool, bool]
        If true -> there is a solid frame around 
        the domain. Important for periodic 
        mean width computation.
    crosssection : string
        mean : get a mean width/length/height per crosssection
        min  : get the min w/l/h per crosssection
        different : stay with one w/h or l/ ratio per column;
                    this means different w/h ratios per 
                    crosssection 
    cs_weight : bool 
        weight the w/h ratio by the relative area. This is introduced
        since we approximate arbitrary cross-sections by rectangles. 
        For higher perimeter-to-area ratios this results in an error which 
        is reduces by this weight. 

    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    lambda1 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in main direction (z, 2nd dir of 2d array, main pressure gradient)
    lambda2 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in perpendicular direction 
    
    Factor Maps here are only based on 'channel' ratios.
    
    """

    # h_Omega
    vox_per_height = geom.shape[0]
    h_map_scaled = ld.maps_and_distances.get_hmap(geom, voxelsize, vox_per_height)
    h_map01 = ld.maps_and_distances.unscale_hmap(h_map_scaled, voxelsize * vox_per_height)

    # Get labels per cross-section
    col_labels, row_labels = ld.maps_and_distances.label_2d_geom(h_map_scaled, solidframe) 

    # Comes in voxels, therefore we use h_map01 * vox_per_height to divide 
    # Results in same ratios as if it would in real dimensions
    # weights store the max w and l values temporarily
    w_map, l_map = ld.maps_and_distances.get_wl_maps_3d(geom)
    mean_w_map, mean_l_map, w_weight, l_weight = ld.maps_and_distances.get_wl_per_stack_3d(w_map, l_map, channelwidth)

    if crosssection == 'mean' or crosssection == 'min':
        # Assign values per cross-section
        mean_w_map     = ld.maps_and_distances.assign_values(col_labels, mean_w_map, crosssection)
        mean_l_map     = ld.maps_and_distances.assign_values(row_labels, mean_l_map, crosssection)
        mean_h_map_col = ld.maps_and_distances.assign_values(col_labels, h_map01 * vox_per_height, crosssection)
        mean_h_map_row = ld.maps_and_distances.assign_values(row_labels, h_map01 * vox_per_height, crosssection)
        w_cs_max       = ld.maps_and_distances.assign_values(col_labels, np.multiply(w_weight, h_map01 * vox_per_height), 'max')
        l_cs_max       = ld.maps_and_distances.assign_values(row_labels, np.multiply(l_weight, h_map01 * vox_per_height), 'max')
    elif crosssection == 'different':
        mean_w_map     = mean_w_map
        mean_l_map     = mean_l_map
        mean_h_map_col = h_map01* vox_per_height
        mean_h_map_row = h_map01* vox_per_height
        w_cs_max       = np.multiply(w_weight, h_map01 * vox_per_height)
        l_cs_max       = np.multiply(l_weight, h_map01 * vox_per_height)
    else:
        raise ValueError('No valid method defined for crosssection!')

    # print(f'Assigned mean values: shape: {mean_w_map[2:-2].shape}, w: {np.mean(mean_w_map[2:-2])}, l: {np.mean(mean_l_map[2:-2])}')


    # Compute ratios
    wh_ratio = np.divide(mean_w_map, mean_h_map_col, out=np.zeros_like(mean_w_map), where=mean_h_map_col!=0)
    lh_ratio = np.divide(mean_l_map, mean_h_map_row, out=np.zeros_like(mean_l_map), where=mean_h_map_row!=0)

    if cs_weight == True:
        mean_w_weight = np.divide(w_cs_max, np.multiply(mean_w_map, mean_h_map_col))
        mean_l_weight = np.divide(l_cs_max, np.multiply(mean_l_map, mean_h_map_row))
        wh_ratio = np.multiply(wh_ratio, mean_w_weight)
        lh_ratio = np.multiply(lh_ratio, mean_l_weight)

    # Actual factor computation
    lambda1 = ld.maps_and_distances.apply_empirical_wh_relation(wh_ratio)
    lambda2 = ld.maps_and_distances.apply_empirical_wh_relation(lh_ratio)
    
    # Normalize factor maps
    lambda1 = np.divide(lambda1, ld.constants.DUMUX_PREFACTOR)
    lambda2 = np.divide(lambda2, ld.constants.DUMUX_PREFACTOR)

    return h_map_scaled, lambda1, lambda2


def lambda_gi_map(h_map01, voxelsize, height, smooth = False, sigma = 0.0):
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
    smooth : bool 
        Smooth the gradient before factor computation
    sigma : float
        Standard deviation for Gaussian kernel.
    
    Returns
    -------
    lambda1 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in main direction (z, 2nd dir of 2d array, main pressure gradient)
    lambda2 : numpy ndarray
        2d map containing factors to use in DUMUX 
        pseudo-3D Stokes simulator in perpendicular direction 
    
    Factor Maps here are only based on gradient.

    """

    # h_Omega
    h_map_scaled = ld.maps_and_distances.scale_hmap(h_map01, height)
    h_map_scaled = ld.maps_and_distances.h_map_settle_rounding_error(h_map_scaled, voxelsize)

    # Compute gradient
    grad_main = ld.wrap_math.grad2d(h_map_scaled, 1)[1]/voxelsize
    grad_perp = ld.wrap_math.grad2d(h_map_scaled, 1)[0]/voxelsize

    if smooth:
        grad_main = smooth_factor_map(grad_main, sigma)
        grad_perp = smooth_factor_map(grad_perp, sigma)

    # Actual factor computation
    lambda1 = ld.maps_and_distances.apply_empirical_grad_relation(np.abs(grad_main))
    lambda2 = ld.maps_and_distances.apply_empirical_grad_relation(np.abs(grad_perp))
    
    # Normalize factor maps
    lambda1 = np.divide(lambda1, ld.constants.DUMUX_PREFACTOR)
    lambda2 = np.divide(lambda2, ld.constants.DUMUX_PREFACTOR)

    # Adjust for skew gradient computation
    lambda1[:, 0] = lambda2[: ,1]
    lambda2[0, :] = lambda2[1, :]

    return lambda1, lambda2



def lambda_gi_map_3d(geom, voxelsize, smooth = False, sigma = 0.0):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.
    voxelsize : float [ m ]
        Voxelsize of the domain.
    smooth : bool 
        Smooth the gradient before factor computation
    sigma : float
        Standard deviation for Gaussian kernel.
    
    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    lambda1 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in main direction (z, 2nd dir of 2d array, main pressure gradient)
    lambda2 : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in perpendicular direction 
    
    Factor Maps here are only based on gradient.
    Same as in 2d, since gradient is 2d.

    """

    vox_per_height = geom.shape[0]

    h_map_scaled = ld.maps_and_distances.get_hmap(geom, voxelsize, geom.shape[0])
    h_map_scaled = ld.maps_and_distances.h_map_settle_rounding_error(h_map_scaled, voxelsize)
    h_map01      = ld.maps_and_distances.unscale_hmap(h_map_scaled, voxelsize * vox_per_height)

    lambda1, lambda2 = lambda_gi_map(h_map01, voxelsize, vox_per_height, smooth = smooth, sigma = sigma)

    return h_map_scaled, lambda1, lambda2


def lambda_total_map(h_map01, voxelsize, height, channelwidth, solidframe, smooth, sigma, crosssection, cs_weight):
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
    smooth : bool 
        Smooth the gradient before factor computation
    sigma : float
        Standard deviation for Gaussian kernel.
    crosssection : string
        mean : get a mean width/length/height per crosssection
        min  : get the min w/l/h per crosssection
        different : stay with one w/h or l/ ratio per column;
                    this means different w/h ratios per 
                    crosssection 
    cs_weight : bool 
        weight the w/h ratio by the relative area. This is introduced
        since we approximate arbitrary cross-sections by rectangles. 
        For higher perimeter-to-area ratios this results in an error which 
        is reduces by this weight. 

    Returns
    -------
    lambda1_total : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in main direction (z, 2nd dir of 2d array, main pressure gradient)
    lambda2_total : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in perpendicular direction 

    """

    # h_Omega
    h_map_scaled = ld.maps_and_distances.scale_hmap(h_map01, height)
    h_map_scaled = ld.maps_and_distances.h_map_settle_rounding_error(h_map_scaled, voxelsize)

    # Compute singular maps
    lambda1_wh, lambda2_wh = lambda_wh_map(h_map01, voxelsize, height, channelwidth, solidframe, crosssection, cs_weight)
    lambda1_gi, lambda2_gi = lambda_gi_map(h_map01, voxelsize, height, smooth, sigma)
    h_Omega_by_hx = ld.maps_and_distances.get_hratio_map(h_map_scaled, voxelsize, height)

    lambda1_total = lambda1_wh * lambda1_gi * h_Omega_by_hx
    lambda2_total = lambda2_wh * lambda2_gi * h_Omega_by_hx

    return lambda1_total, lambda2_total


def lambda_total_map_3d(geom, voxelsize, channelwidth, solidframe, crosssection, smooth, sigma, cs_weight):
    """
    
    Parameters
    ----------
    geom : numpy.ndarray
        3d domain of the porous material.
        Zero -> indicates fluid voxel.
        One  -> indictaes solid voxel.
    voxelsize : float [ m ]
        Voxelsize of the domain.
    channelwidth : string
        mean : return mean channel width per column
        min  : return min channel width per column
        harmonic : return harmonic mean channel width per column
    solidframe: list [bool, bool]
        If true -> there is a solid frame around 
        the domain. Important for periodic 
        mean width computation.
    smooth : bool 
        Smooth the gradient before factor computation
    sigma : float
        Standard deviation for Gaussian kernel.
    crosssection : string
        mean : get a mean width/length/height per crosssection
        min  : get the min w/l/h per crosssection
        different : stay with one w/h or l/ ratio per column;
                    this means different w/h ratios per 
                    crosssection 
    cs_weight : bool 
        weight the w/h ratio by the relative area. This is introduced
        since we approximate arbitrary cross-sections by rectangles. 
        For higher perimeter-to-area ratios this results in an error which 
        is reduces by this weight. 

    Returns
    -------
    h_map_scaled :numpy ndarray
        2d, height of geometry
        scaled by the voxelsize
    lambda1_total : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in main direction (z, 2nd dir of 2d array, main pressure gradient)
    lambda2_total : numpy ndarray
        2d map containing factors to use in DUMUX simulator
        in perpendicular direction 
    
    """
    vox_per_height = geom.shape[0]

    h_map_scaled,  lambda1_wh, lambda2_wh = lambda_wh_map_3d(geom, voxelsize, channelwidth, solidframe, crosssection, cs_weight)
    _h_map_scaled, lambda1_gi, lambda2_gi = lambda_gi_map_3d(geom, voxelsize, smooth, sigma)

    h_Omega_by_hx = ld.maps_and_distances.get_hratio_map(h_map_scaled, voxelsize, vox_per_height)

    lambda1_total = lambda1_wh * lambda1_gi * h_Omega_by_hx
    lambda2_total = lambda2_wh * lambda2_gi * h_Omega_by_hx

    return h_map_scaled, lambda1_total, lambda2_total

