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
import os, glob
import scipy
import cv2
import stat
import matplotlib.pyplot as plt

import localdrag as ld

###--------------------------------------------------------------------------------

def all_limits(shape, DD_TYPE, DECOMP, rank, verbose = False):
    """
    """

    if DD_TYPE == 'fixed':
        ny = DECOMP[0]
        nx = DECOMP[1]
    elif DD_TYPE == 'rect_main':
        nx = DECOMP
        approx_size = shape[1]/nx
        ny = int(round(shape[0]/approx_size))
    elif DD_TYPE == 'rect_perp':
        ny = DECOMP
        approx_size = shape[0]/ny
        nx = int(round(shape[1]/approx_size))

    if verbose == True:
        if rank == 0:
            print(f'Rank {rank}: Number of subdomains [{ny}, {nx}]')
            print(f'Rank {rank}: Ratio edge length: {(shape[0]/ny)/(shape[1]/nx)}')

    lx = (np.linspace(0, shape[1], shape[1]+1, dtype = int)).tolist()
    kx, mx = divmod(shape[1], nx)
    xranges = list(lx[i*kx+min(i, mx):(i+1)*kx+min(i+1, mx)] for i in range(nx))
    xstarts = []
    xends = []
    for i in range(len(xranges)):
        xstarts.append(xranges[i][0])
        xends.append(xranges[i][-1])

    ly = (np.linspace(0, shape[0], shape[0]+1, dtype = int)).tolist()
    ky, my = divmod(shape[0], ny)
    yranges = list(ly[i*ky+min(i, my):(i+1)*ky+min(i+1, my)] for i in range(ny))
    ystarts = []
    yends = []
    for i in range(len(yranges)):
        ystarts.append(yranges[i][0])
        yends.append(yranges[i][-1])

    if verbose == True:
        if rank == 0:
            print(f'Rank {rank}: shape {shape}')
            print(f'Rank {rank}: ystarts {ystarts}')
            print(f'Rank {rank}: yends {yends}')
            print(f'Rank {rank}: xstarts {xstarts}')
            print(f'Rank {rank}: xends {xends}')

    all_limits = {}

    # j/y -> rows 
    # i/x -> columns
    for i in range(nx):
        for j in range(ny):
            all_limits[f'{j:02}_{i:02}'] = [[ystarts[j], yends[j]], [xstarts[i], xends[i]]]

    return all_limits


def all_limits_REV(MIN_VOX, STEP_VOX, shape):
    """
    """

    all_limits = {}
    number_domains = int(np.floor((np.min(shape) - MIN_VOX) / STEP_VOX + 1))

    for i in range(number_domains):
        size = MIN_VOX + i * STEP_VOX
        # Upper left corner
        c1_xstart = 0
        c1_ystart = 0
        c1_xend   = size
        c1_yend   = size

        # Lower left corner
        c2_xstart = 0
        c2_ystart = shape[0] - size
        c2_xend   = size
        c2_yend   = shape[0] 

        # Upper right corner 
        c3_xstart = shape[1] - size
        c3_ystart = 0
        c3_xend   = shape[1]
        c3_yend   = size 

        # Lower right corner
        c4_xstart = shape[1] - size
        c4_ystart = shape[0] - size
        c4_xend   = shape[1]
        c4_yend   = shape[0] 

        all_limits[f'C1_{int(size)}'] = [[c1_ystart, c1_yend], [c1_xstart, c1_xend]]
        all_limits[f'C2_{int(size)}'] = [[c2_ystart, c2_yend], [c2_xstart, c2_xend]]
        all_limits[f'C3_{int(size)}'] = [[c3_ystart, c3_yend], [c3_xstart, c3_xend]]
        all_limits[f'C4_{int(size)}'] = [[c4_ystart, c4_yend], [c4_xstart, c4_xend]]

    return all_limits



# def keys_per_rank(all_limits, nprocs, rank):
#     """
#     """


#     mydomains = (np.linspace(0, len(all_limits), len(all_limits)+1, dtype = int)).tolist()
#     k, m = divmod(len(all_limits), nprocs)
#     ranges = list(mydomains[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(nprocs))

#     all_keys = list(all_limits.keys())

#     # Check this for number of keys is smaller than nprocs
#     mykeys = all_keys[ranges[rank][0]:ranges[rank][-1]+1]

#     return mykeys


def distribute_tasks(filelist, nprocs, rank):
    """
    """
    mydomains = (np.linspace(0, len(filelist), len(filelist)+1, dtype = int)).tolist()
    k, m = divmod(len(filelist), nprocs)
    ranges = list(mydomains[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(nprocs))

    if not ranges[rank]:
        myfiles = []
    else:
        myfiles = filelist[ranges[rank][0]:ranges[rank][-1]+1]

    return myfiles 


def create_bash_file(outfolder, parallel = True, cores = 8):
    """
    """
    runfilename = 'runSimulations.sh'
    currentdir = os.getcwd()
    os.chdir(outfolder)
    subfolders = glob.glob(f'./subdomains*/**/')
    
    if os.path.isfile(runfilename):
        os.remove(runfilename)

    header = '#!/bin/bash'
    with open(runfilename, "w") as f:
        f.write(f'{header}\n\n')
        for i in range(len(subfolders)):
            if parallel == False:
                mystr = f'python3 multiprocrunStokesGeneric_all.py -dir {subfolders[i]} -pfGiven'
            else:
                mystr = f'python3 multiprocrunStokesGeneric_all.py -dir {subfolders[i]} -pfGiven -parallel'

            f.write(f'{mystr}\n')

    # make bash file exec
    status = os.stat(runfilename)
    os.chmod(runfilename, status.st_mode | stat.S_IEXEC)
    os.chdir(currentdir)
