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

argParser = argparse.ArgumentParser()
argParser.add_argument("-dir",      "--rawDirectory",             help="Path to directory where .raw files are located.")
argParser.add_argument("-vs",       "--voxelSize",    type=float, help="Defines the voxel size of the image in [m].")
argParser.add_argument("-v_height", "--voxperheight", type=float, help="Defines the number of voxels per height of the domain [m].")
args = argParser.parse_args()


datadir        = args.rawDirectory
vox_per_height = args.voxperheight
voxelsize      = args.voxelSize
height         = voxelsize * vox_per_height

for file in glob.glob(f'{datadir}/*.raw'):
    print("Processing file: " + file)
    filename = file
    size = wrap_import.getNumVoxelFromName(filename)
    voxelsize = wrap_import.getVoxelSizeFromName(filename)

    # Remove the upper and lower solid frame
    geom = wrap_import.read_3d_rawfile(filename, size, voxelsize)
    geom, size = wrap_import.remove_frame(geom, axis = 0)
    geom, size = wrap_import.remove_frame(geom, axis = 1)

    h_map01 = evaluate3d.get_hmap01(geom)
    h_map_scaled = maps_and_distances.scale_hmap(h_map01, height)

    fn = filename.replace(f'{datadir}/', '')
    fn = fn.replace('.raw', '')

    write_maps.write2pgm('./', f'hx_{fn}', h_map_scaled)

