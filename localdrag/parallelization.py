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
import argparse

import localdrag as ld

###--------------------------------------------------------------------------------


def distribute_args_REV_series(comm, rank):
    """
    """
    argParser = argparse.ArgumentParser()
    argParser.add_argument( "-dir",      "--datadir",                                              help="Working directory with .pgm files."                              )
    argParser.add_argument( "-min_vox",  "--min_vox",          default=None,  help="Start value for voxel number."                                   )
    argParser.add_argument( "-v_step",   "--vox_step",         default=None,  help="Number of voxels per step."                                      )
    argParser.add_argument( "-v_size",   "--voxelsize",        default=None,  help="Voxelsize in meter."                                             )
    argParser.add_argument( "-height",   "--height",           default=None,  help="Height of domain in meter."                                      )
    argParser.add_argument( "-solidfx",  "--solidframe_x",     default=False, help="Solid Frame in x direction."                                     )
    argParser.add_argument( "-solidfy",  "--solidframe_y",     default=False, help="Solid Frame in y direction."                                     )
    argParser.add_argument( "-sigma",    "--sigma",            default=0.0,   help="Smoothing length for prefactor computation."                     )
    argParser.add_argument( "-smooth",   "--smooth",           default=0.0,   help="Smoothing applied."                     )
    argParser.add_argument( "-csw",      "--cs_weight",        default=False, help="Weighting crosssection skewness."                                )
    argParser.add_argument( "-cw",       "--channelwidth",     default='mean',help="Method for channelwitdh."                                        )
    argParser.add_argument( "-cs",       "--crosssection",     default='mean',help="Method for crosssection."                                        )
    argParser.add_argument( "-verbose",  "--verbose",          default=False ,help="Verbose mode."                                                   )
    argParser.add_argument( "-plt",      "--plot",             default=False ,help="Plot results."                                                   )

    # Initialize here for non-zero ranks
    args = None 

    try:
        if rank == 0:
            args = argParser.parse_args()
    # Execute not matter the try results
    finally:
        args = comm.bcast(args, root = 0)

    # Clean exit
    if args is None:
        exit(0)

    return args 