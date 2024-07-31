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
import shutil
import time

import localdrag as ld

###--------------------------------------------------------------------------------


def update_input_parameter_file(folder, myfilestr, md):
    """
    
    Parameters
    ----------
    folder : string
        Path to file directory with input data.
    myfilestr : string
        output file name.
    md : dict
        Collected md.

    Returns
    -------
    Nothing.
    
    Put relevant Meta-Parameter in Dumux input file template

    """

    hx_file = f'{myfilestr}.pgm'

    print(hx_file)

    if md['lambda_given']:
        lambda1_file = myfilestr.replace('hx', 'lambda1')
        lambda1_file = f'{lambda1_file}.txt'
        lambda2_file = myfilestr.replace('hx', 'lambda2')
        lambda2_file = f'{lambda2_file}.txt'
    else:
        pfMapMain = 'noFile'
        pfMapPerp = 'noFile'


    # cellsize  = ld.wrap_import.getNumVoxelFrom2DName( myfilestr )
    _array, cellsize  = ld.wrap_import.read_pgm(f'{folder}/{hx_file}')
    cells_x   = str( cellsize[1] )
    cells_y   = str( cellsize[0] )
    size_x    = str(round( cellsize[1] * float( md['voxelsize'] ), 6 ) ) # main pressure gradient direction
    size_y    = str(round( cellsize[0] * float( md['voxelsize'] ), 6 ) ) # perp direction

                
    text_to_search_list = [ 
                            '<geomFile>', 
                            '<name>', 
                            '<domainSizeX>', '<domainSizeY>', 
                            '<cellsX>', '<cellsY>',
                            '<PreFactorDragFileX>', '<PreFactorDragFileY>',
                            '<WriteVtuData>', 
                            '<height>', 
                            '<deltaP>'
                          ]
    
    params              = [ 
                            hx_file, 
                            myfilestr, 
                            size_x, size_y, 
                            cells_x, cells_y, 
                            lambda1_file, lambda2_file,
                            str( md['vtuOutput'] ),
                            str( md['height'] ), 
                            str( md['pressure'] )
                          ]
    
    # print(f'Params: {params}')

    with open(f'{folder}/{myfilestr}.input', 'r') as f:
        filedata = f.read()

    # Replace the target string
    for text_to_search, param in zip(text_to_search_list, params):
        filedata = filedata.replace(text_to_search, param)   
    
    with open(f'{folder}/{myfilestr}.input', 'w') as f:
        f.write(filedata)



def run_dumux(datadir, executable, input_file_name, output_file_name):
    """
    
    Parameters
    ----------
    datadir : string
        Path to file directory with input data.
    executable : string
        Dumux executable.
    input_file_name : string
        Name of Dumux input file.
    output_file_name : string
        Name of Dumux output/log file.

    Returns
    -------
    Nothing.
    

    """

    commandstr = f'./{executable} {input_file_name} >> {output_file_name}'

    cwd = os.getcwd()
    os.chdir(datadir)

    if os.path.isfile(output_file_name):
        os.remove(output_file_name)

    # Execute dumux simulation
    os.system(commandstr)
    os.chdir(cwd)



def get_executeable(md):
    """
    
    Parameters
    ----------
    md : dict
        Dictionary containing simulation metadata.

    Returns
    -------
    Nothing.
    

    """
    if md['copy_run']:
        shutil.copy2(f'{md["dumuxpath"]}/{md["executable"]}', f'{md["datadir"]}/{md["executable"]}')
        if not os.access(f'{md["datadir"]}/{md["executable"]}', os.X_OK):
            # get access privileges
            status = os.stat(f'{md["datadir"]}/{md["executable"]}')
            # add x to access privilege for u
            os.chmod(f'{md["datadir"]}/{md["executable"]}', status.st_mode | stat.S_IEXEC)


def run(metadata, myfilestr):
    """
    
    Parameters
    ----------
    metadata : dict
        Dictionary containing simulation metadata.
    myfilestr : string
        output file name.

    Returns
    -------
    Nothing.
    

    """
    
    # reverse  
    datadir            = metadata['datadir']
    generic_input_file = metadata['generic_input_file']
    lambda_given       = metadata['lambda_given']
    verbose            = metadata['verbose']
    copy_run           = metadata['copy_run']
    executable         = metadata['executable']
    parallel           = metadata['parallel']
    
    if parallel:
        thread = multiprocessing.current_process()._identity[0]
        print(f'Thread {thread}: {myfilestr}')
    else:
        print(f'Computing: {myfilestr}')

    # string operations to get filenames
    myfilestr = myfilestr.replace('.pgm', '')
    myfilestr = myfilestr.replace(f'{datadir}/', '')

    input_file_name  = f'{myfilestr}.input'
    output_file_name = f'{myfilestr}.output'


    # copy and run 
    if copy_run:
        shutil.copyfile(generic_input_file, f'{datadir}/{input_file_name}')
        update_input_parameter_file(datadir, myfilestr, metadata)

        # Start timer for simulation
        start = time.time()

        if verbose:
            if parallel:
                print(f'Thread {thread}: Starting simulation.')
            else:
                print(f'Starting simulation.')

        # actual dumux simulation wrapper
        run_dumux(datadir, executable, input_file_name, output_file_name)
        elapsed_time = time.time() - start

        if parallel:
            print(f'Thread {thread}: Simulation took {(elapsed_time/60):2.2f} minutes resp. {(elapsed_time):2.2f} seconds')
        else:
            print(f'Simulation took {(elapsed_time/60):2.2f} minutes resp. {(elapsed_time):2.2f} seconds')


