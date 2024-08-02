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

argParser = argparse.ArgumentParser()
argParser.add_argument("-dir",       "--pgmDirectory",                                                      help="Path to directory where .pgm files are located.")
argParser.add_argument("-vs",        "--voxelSize",       type=float,                                       help="Defines the voxel size of the image in [m].")
argParser.add_argument("-height",    "--heightOfDomain",  type=float,                                       help="Defines the height of the domain [m].")
argParser.add_argument("-pltRes",    "--plotResults",                 action='store_true', default=False,   help="Plot results of the lambda values should be stored as a png file, default is False.")
argParser.add_argument("-method",    "--methodLambda",                                     default="total", help="Method which is used to calculate the lambda fields. Options are 'wh_only', 'grad_only', 'total', or 'all' which creates the lambda values for all types in separate folders. The default method is 'total'")
argParser.add_argument("-minw",      "--minimum_w",                   action='store_true', default=False,   help="Use the minimum width for all ew ratios. Default is False.")
argParser.add_argument("-sfx",       "--solidframex",                 action='store_true', default=False,   help="Actual domain has a solid frame in x direction. Important for computation of channel width. If False, periodicity is assumed. Default is False.")
argParser.add_argument("-sfy",       "--solidframey",                 action='store_true', default=False,   help="Actual domain has a solid frame in y direction. Important for computation of channel width. If False, periodicity is assumed. Default is False.")
argParser.add_argument("-sigma",     "--sigma",           type=float,                      default=0.0,     help="Standard deviation for Gaussian kernel. if smoothing the gradient is switched on.")
argParser.add_argument("-smooth",    "--smooth",                      action='store_true', default=False,   help="Smooth the gradient before factor computation. Default is False")
argParser.add_argument("-csweight",  "--csweight",                    action='store_true', default=False,   help="Weight the w/h ratio by the relative area. This is introduced since we approximate arbitrary cross-sections by rectangles. For higher perimeter-to-area ratios this results in an error which is reduces by this weight. Default is False.")
argParser.add_argument("-cw",        "--channelwidth",    type=str,                        default="mean",  help="Averaging method for the channelwidth. Default is mean. Also possible: harmonic or min.")
argParser.add_argument("-cs",        "--crosssection",    type=str,                        default="mean",  help="Averaging method for crosssection values. Default is mean. Also possible: min or different.")


args = argParser.parse_args()

pgmDirectory = args.pgmDirectory
voxelsize    = args.voxelSize
height       = args.heightOfDomain
plotResults  = args.plotResults
methodInput  = args.methodLambda
MINW         = args.minimum_w
solidframe   = list([args.solidframex, args.solidframey])
sigma        = args.sigma
smooth       = args.smooth
cs_weight    = args.csweight
channelwidth = args.channelwidth
crosssection = args.crosssection

all_files = natsorted(glob.glob(pgmDirectory + "/*.pgm"))

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

    for method in ['wh_only', 'grad_only', 'total']:
        if methodInput != 'all' and methodInput != method:
            continue
        current_directory = os.getcwd()
        outpath = f'{current_directory}/{pgmDirectory.split("/")[-1]}_{method}'
        os.system(f'mkdir -p {outpath}')

        print(f'Used method: {method}')
        # Scale hmap with h_Omega
        h_map_scaled = ld.maps_and_distances.scale_hmap(h_map01, height)

        if method == 'grad_only':
            lambda1, lambda2 = ld.create_lambda_maps.lambda_gi_map(h_map01, voxelsize, height, smooth, sigma)
        elif method == 'wh_only':
            lambda1, lambda2 = ld.create_lambda_maps.lambda_wh_map(h_map01, voxelsize, height, channelwidth, solidframe, crosssection, cs_weight)
        elif method == 'total':
            lambda1, lambda2 = ld.create_lambda_maps.lambda_total_map(h_map01, voxelsize, height, channelwidth, solidframe, smooth, sigma, crosssection, cs_weight)
        else:
            raise ValueError('No valid method defined.')

        # Cut the domains for Dumux pseudo-3D solver
        # Does not need solid frame to realize no-slip BC
        h_map_scaled, size = ld.write_maps.remove_temp_frame(h_map_scaled)
        lambda1, size = ld.write_maps.remove_temp_frame(lambda1)
        lambda2, size = ld.write_maps.remove_temp_frame(lambda2)

        fn = filename.split('/')[-1]
        fn = fn.replace('.pgm', '')
        
        # Write the domains for Dumux
        ld.write_maps.write2txt(outpath, fn, 'lambda1', lambda1)
        ld.write_maps.write2txt(outpath, fn, 'lambda2', lambda2)
        ld.write_maps.write2pgm(outpath, f'hx_{fn}', h_map_scaled)

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
