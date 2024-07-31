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
import argparse
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
going to be collpased. This is different in Dumux pseudo-3D solver, 
where the main pressure gradient direction always is e_1 similar 
as in the paper. 

Naming of output files:
main -> main pressure gradient direction
perp -> perpendicular pressure gradient direction

"""

#plotResults = False
#vox_per_height = 36
MINW = False
solidframe = [False, True] # why is second entry true?
sigma = 0.0
smooth = False
cs_weight = False
channelwidth = 'mean'
crosssection = 'mean'
#voxelsize = 1e-6 #[m]
#folderPath = "../pgmfiles_new"
### --------------------------

argParser = argparse.ArgumentParser()
argParser.add_argument("-dir", "--workingDirectory", help="working directory with .pgm files")
argParser.add_argument("-vs", "--voxelSize", type=float, help="defines the voxel size of the image in [m]")
argParser.add_argument("-height", "--heightOfDomain", type=float, help="defines the height of the domain [m]")
argParser.add_argument("-pltRes", "--plotResults", action='store_true', default=False, help="Defines if the results of the lambda values should be stored as a png file, default is False")
argParser.add_argument("-method", "--methodLambda", default = "total", help="defines the the method which is used to calculate the lambda methods. Option are 'wh_only', 'grad_only', 'total', or 'all' which creates the lambda values for all types in separate folders. The default method is 'total' ")

args = argParser.parse_args()

workingDirectory = args.workingDirectory + "/"
voxelsize = args.voxelSize
height = args.heightOfDomain
plotResults = args.plotResults
methodInput = args.methodLambda

# this is not nice - we shouldn't the parameter vox_per_height in the d2to2d option, if possible
vox_per_height = round(height/voxelsize)

all_files = natsorted(glob.glob(workingDirectory + "*.pgm"))

# loop through all .raw files in the folder and create h_map, lambda1, lambda2
for file in all_files:
    print("Processing file: " + file)
    filename = file
    
    h_map01, size_ = ld.wrap_import.read_pgm(filename)
    # The single precipitate domain lives in an solid channel
    # that is not required for Dumux pseudo-3D solver, as 
    # no-slip BC are setup, without it. We need it in the pre-
    # processing, since otherwise the module assumes 
    # periodicity for the channel width.
    h_map01 = ld.wrap_import.h_map_frame(h_map01)
    
    #voxelsize = ld.wrap_import.getVoxelSizeFromName(filename)

    for method in ['wh_only', 'grad_only', 'total']:
        if methodInput != 'all' and methodInput != method:
            continue
        outpath = workingDirectory + method
        os.system(f'mkdir -p {outpath}')
        #outpath = f'2d_{method}/'
        print(f'Used method: {method}')
        # Scale hmap with h_Omega
        #height = voxelsize * vox_per_height
        h_map_scaled = ld.maps_and_distances.scale_hmap(h_map01, height)

        if method == 'grad_only':
            lambda1, lambda2 = ld.create_lambda_maps.lambda_gi_map(h_map01, voxelsize, vox_per_height, smooth, sigma)
        elif method == 'wh_only':
            lambda1, lambda2 = ld.create_lambda_maps.lambda_wh_map(h_map01, voxelsize, vox_per_height, channelwidth, solidframe, crosssection, cs_weight)
        elif method == 'total':
            lambda1, lambda2 = ld.create_lambda_maps.lambda_total_map(h_map01, voxelsize, vox_per_height, channelwidth, solidframe, smooth, sigma, crosssection, cs_weight)
        else:
            raise ValueError('No valid method defined.')

        # Cut the domains for Dumux pseudo-3D solver
        # Does not need solid frame to realize no-slip BC
        h_map_scaled, size = ld.write_maps.remove_temp_frame(h_map_scaled)
        lambda1, size = ld.write_maps.remove_temp_frame(lambda1)
        lambda2, size = ld.write_maps.remove_temp_frame(lambda2)

        fn = filename.replace(workingDirectory, '')
        fn = fn.replace('.pgm', '')
        
        print("outpath: " + outpath)
        print("fn: " + fn)
        # Write the domains for Dumux
        ld.write_maps.write2txt(outpath, fn, 'lambda1', lambda1)
        ld.write_maps.write2txt(outpath, fn, 'lambda2', lambda2)
        ld.write_maps.write2pgm(outpath, fn, 'hx', h_map_scaled)

        if plotResults:
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
