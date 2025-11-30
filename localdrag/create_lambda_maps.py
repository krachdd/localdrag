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

def wh(h_map01, voxelsize, height, channelwidth, solidframe, crosssection, cs_weight, solver = 'stokes'):
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
    lambda1 = ld.maps_and_distances.apply_empirical_wh_relation(wh_ratio, solver)
    lambda2 = ld.maps_and_distances.apply_empirical_wh_relation(lh_ratio, solver)
    
    # Normalize factor maps
    lambda1 = np.divide(lambda1, ld.constants.DUMUX_PREFACTOR)
    lambda2 = np.divide(lambda2, ld.constants.DUMUX_PREFACTOR)

    return lambda1, lambda2



def gi(h_map01, voxelsize, height, smooth = False, sigma = 0.0, solver = 'stokes'):
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
    lambda1 = ld.maps_and_distances.apply_empirical_grad_relation(np.abs(grad_main), solver)
    lambda2 = ld.maps_and_distances.apply_empirical_grad_relation(np.abs(grad_perp), solver)
    
    # Normalize factor maps
    lambda1 = np.divide(lambda1, ld.constants.DUMUX_PREFACTOR)
    lambda2 = np.divide(lambda2, ld.constants.DUMUX_PREFACTOR)

    # Adjust for skew gradient computation
    lambda1[:, 0] = lambda2[: ,1]
    lambda2[0, :] = lambda2[1, :]

    return lambda1, lambda2



def p(h_map01, voxelsize, height, smooth = False, sigma = 0.0):
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
    lambda1 = ld.maps_and_distances.apply_empirical_pcorrect_relation(np.abs(grad_main))
    lambda2 = ld.maps_and_distances.apply_empirical_pcorrect_relation(np.abs(grad_perp))
    
    # Normalize factor maps
    lambda1 = np.divide(lambda1, ld.constants.DUMUX_PREFACTOR)
    lambda2 = np.divide(lambda2, ld.constants.DUMUX_PREFACTOR)

    # Adjust for skew gradient computation
    lambda1[:, 0] = lambda2[: ,1]
    lambda2[0, :] = lambda2[1, :]

    return (lambda1 - lambda2) * grad_main, (lambda1 - lambda2) * grad_perp


def total(h_map01, voxelsize, height, channelwidth, solidframe, smooth, sigma, crosssection, cs_weight, solver = 'stokes'):
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
    solver : str
        Solver for balance of linear momentum used in Dumux or other software.

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
    if solver != 'height_averaged':
        lambda1_wh, lambda2_wh = wh(h_map01, 
                                    voxelsize, 
                                    height, 
                                   channelwidth, 
                                   solidframe, 
                                   crosssection, 
                                   cs_weight,
                                   solver)
        lambda1_gi, lambda2_gi = gi(h_map01, 
                                   voxelsize, 
                                   height, 
                                   smooth, 
                                   sigma,
                                   solver)

    if solver == 'stokes':
        h_Omega_by_hx = ld.maps_and_distances.hratio_map(h_map_scaled, voxelsize, height)
        lambda1_total = lambda1_wh * lambda1_gi * h_Omega_by_hx
        lambda2_total = lambda2_wh * lambda2_gi * h_Omega_by_hx
    elif solver == 'brinkman':
        lambda1_total = lambda1_wh * lambda1_gi
        lambda2_total = lambda2_wh * lambda2_gi
    elif solver == 'analytical_brinkman':
        Fgrad11_h1, Fgrad12_h1, Fgrad21_h1, Fgrad22_h1 = ld.maps_and_distances.Fgrad_map(h_map_scaled, voxelsize, height)
        lambda1_total = 0.5 * lambda1_wh * ( 1.0 + ( lambda1_gi * Fgrad11_h1 + lambda2_gi * Fgrad12_h1 ))
        lambda2_total = 0.5 * lambda2_wh * ( 1.0 + ( lambda1_gi * Fgrad21_h1 + lambda2_gi * Fgrad22_h1 ))
    elif solver == 'height_averaged':
        Fgrad11_h1, Fgrad12_h1, Fgrad21_h1, Fgrad22_h1 = ld.maps_and_distances.Fgrad_map(h_map_scaled, voxelsize, height)
        lambda1_total = (Fgrad11_h1 + Fgrad12_h1 + 1.) * 0.5
        lambda2_total = (Fgrad21_h1 + Fgrad22_h1 + 1.) * 0.5

        # if pressure_correction:
        #     raise ValueError('Pressure Correction must not be set if height height_averaged is used!')
    else:
        raise ValueError('Dumux Solver not set correctly!')

    return lambda1_total, lambda2_total


