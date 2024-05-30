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
import os, sys, glob
import matplotlib.pyplot as plt
from natsort import natsorted

sys.path.append('../../')

import localdrag as ld
from localdrag import *

###--------------------------------------------------------------------------------

"""
In input raw file: 
Zero -> indicates fluid voxel
One  -> indictaes solid voxel

In input pgm file:
Zero -> indicates fully solid voxel
One  -> indicates fully fluid voxel

The 3D Stokes Solver poremaps (https://doi.org/10.18419/darus-3676)
introduces a pressure gradient in e_3-direction. e_1-driection is 
going to be collpased.

Naming of output files:
main -> main pressure gradient direction
perp -> perpendicular pressure gradient direction

"""

plotResults = True
MINW = False
solidframe = [False, True]
sigma = 0.0
smooth = False
cs_weight = False
channelwidth = 'mean'
crosssection = 'mean'
### --------------------------

all_files = natsorted(glob.glob("../rawfiles/*.raw"))

# loop through all .raw files in the folder and create h_map, lambda1, lambda2
for file in all_files:
    print("Processing file: " + file)
    filename = file
    size = ld.wrap_import.getNumVoxelFromName(filename)
    voxelsize = ld.wrap_import.getVoxelSizeFromName(filename)
    
    # Remove the upper and lower solid frame
    geom = ld.wrap_import.read_3d_rawfile(filename, size, voxelsize)
    geom, size = ld.wrap_import.remove_frame(geom, axis = 0)

    for method in ['wh_only', 'grad_only', 'total']:
        os.system(f'mkdir -p 2d_{method}')
        outpath = f'2d_{method}/'
        print(f'Used method: {method}')
        if method == 'wh_only':
            h_map_scaled, lambda1, lambda2 = ld.create_lambda_maps.lambda_wh_map_3d(geom, voxelsize, channelwidth, solidframe, crosssection, cs_weight)
        elif method == 'grad_only':
            h_map_scaled, lambda1, lambda2 = ld.create_lambda_maps.lambda_gi_map_3d(geom, voxelsize, smooth, sigma)
        elif method == 'total':
            h_map_scaled, lambda1, lambda2 = ld.create_lambda_maps.lambda_total_map_3d(geom, voxelsize, channelwidth, solidframe, crosssection, smooth, sigma, cs_weight)
        else:
            raise ValueError('No valid method defined.')

        # Cut the domains for Dumux pseudo-3D solver
        # Does not need solid frame to realize no-slip BC
        h_map_scaled, size = ld.wrap_import.remove_frame(h_map_scaled, axis = 0)
        lambda1, size = ld.wrap_import.remove_frame(lambda1, axis = 0)
        lambda2, size = ld.wrap_import.remove_frame(lambda2, axis = 0)

        fn = filename.replace('../rawfiles/', '')
        fn = fn.replace('.raw', '')

        # Write the domains for Dumux
        ld.write_maps.write2txt(outpath, fn, 'lambda1', lambda1)
        ld.write_maps.write2txt(outpath, fn, 'lambda2', lambda2)
        ld.write_maps.write2pgm(outpath, fn, 'hx', h_map_scaled)

        if plotResults:
            # for i in [geom[9, :, :], h_map, lambda1, lambda2]:
            fig, axs = plt.subplots(1, 3, figsize=(20, 4))
            a1 = axs[0].imshow(h_map_scaled)
            a2 = axs[1].imshow(lambda1)
            a3 = axs[2].imshow(lambda2)
            fig.colorbar(a1, ax = axs[0])
            fig.colorbar(a2, ax = axs[1])
            fig.colorbar(a3, ax = axs[2])
            plt.savefig(f'{outpath}{fn}.png')
            # plt.show()
            plt.close()
            plt.clf()
