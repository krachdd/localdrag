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

def seek(output_file_name):
    """
    
    Parameters
    ----------
    output_file_name : str
        file name of the output produced by Dumux
    
    Returns
    -------
    k11 permeability in main pressure gradient direction
    k12 permeability in perpendicular to main pressure gradient direction
    simtime: Simulation time of Dumux simulation

    ...
    k11_darcypermeability: 7.17099e-13
    k12_darcypermeability: -4.91694e-27
    Simulation took 1.74984 seconds on 1 processes.
    ...

    """
    if not os.path.isfile(output_file_name):
        raise FileNotFoundError

    file = open(output_file_name, 'r')
    lines = file.readlines()

    k11 = 0.0
    k12 = 0.0
    simtime = 0.0

    for l in lines:
        if l.strip().startswith('k11_darcypermeability:'):
            k11 = float(l.split(':')[1])
        if l.strip().startswith('k12_darcypermeability:'):
            k12 = float(l.split(':')[1])
        if l.strip().startswith('Simulation took '):
            simtime = float(l.split(' ')[2])

    file.close()

    return k11, k12, simtime


def get_sample(myfilestr):
    """
    
    Parameters
    ----------
    myfilestr : str
        file string of the output produced by Dumux
    
    Returns
    -------
        smaple numbers in both spatial direction

    """

    key = myfilestr.split('sample')[1]
    return int(key.split("_")[0]), int(key.split("_")[-1])


def crawl_output_sweep(filelist, metadata):
    """
    Parameters
    ----------
    filelist : list 
        list of strings with files contaning metadata
    metadata : dictionary 
        containing all metadata of Dumux sims
    
    Returns
    -------
        Nothing

    Crawl log files and save relevant data to txt files.
    """

    datadir = metadata['datadir']

    currentdir = os.getcwd()
    os.chdir(datadir)

    samples_row = []
    samples_col = []
    all_k11     = []
    all_k12     = []
    all_simtime = []

    for i in range(len(filelist)):
        myfilestr = filelist[i]
        myfilestr = myfilestr.replace('.pgm', '')
        myfilestr = myfilestr.replace(datadir, '')

        output_file_name = f'{myfilestr}.output'

        # store sample info
        sample_row, sample_col = get_sample(myfilestr)
        samples_row.append(sample_row)
        samples_col.append(sample_col)

        if not os.path.isfile(output_file_name):
            print(f'File {output_file_name} does not exist')
            continue

        # store permeability info
        k11, k12, simtime = seek(output_file_name)
        all_k11.append(k11)
        all_k12.append(k12)
        all_simtime.append(simtime)

    # concatenate lists
    logdata = np.column_stack((samples_row, samples_col,
                               all_k11, all_k12, 
                               all_simtime))
    logdata = logdata.astype(np.double)
    
    os.chdir(currentdir)

    direction = str(datadir.split('/')[2])
    savedir   = str(datadir.split('/')[1])

    fn = f'{direction}_permeabilities.csv'

    fmt    = '%02d', '%02d', '%.6g', '%.6g', '%1.5f'
    header = 'sample_row, sample_column, k11 [m^2], k12 [m^2], simtime [s]'
    np.savetxt(f'{savedir}/{fn}', logdata, fmt = fmt, header = header, delimiter = ',')


def crawl_output_benchmark(filelist, metadata):
    """
    Parameters
    ----------
    filelist : list 
        list of strings with files contaning metadata
    metadata : dictionary 
        containing all metadata of Dumux sims
    
    Returns
    -------
        Nothing

    Crawl log files and save relevant data to txt files.
    """

    datadir = metadata['datadir']

    currentdir = os.getcwd()
    os.chdir(datadir)

    header = 'file, k11 [m^2], k12 [m^2], simtime [s]\n'

    f = open(f'permeabilities.csv', 'w')
    f.write(header)


    for i in range(len(filelist)):
        myfilestr = filelist[i]
        myfilestr = myfilestr.replace('.pgm', '')
        myfilestr = myfilestr.replace(f'{datadir}/', '')

        output_file_name = f'{myfilestr}.output'

        if not os.path.isfile(output_file_name):
            print(f'File {output_file_name} does not exist')
            continue

        # store permeability info
        k11, k12, simtime = seek(output_file_name)
        f.write(f'{myfilestr}, {k11}, {k12}, {simtime}\n')

    f.close()
    os.chdir(currentdir)



def merge_data(op_str, rp_str, p_str, output_file_name):
    """
    Parameters
    ----------
    op_str : str 
        filename of log file containing information on permeability in org direction 
    rp_str : str 
        filename of log file containing information on permeability in rotated direction 
    p_str : str 
        filename of log file containing information on porosities 
    output_file_name : str 
        filename for merged data 
    
    Returns
    -------
        Nothing

    Crawl log files and save relevant data to txt files.


    Merge all output data that is relevant

    """

    org_perm_samples = np.genfromtxt(op_str, delimiter=',', dtype='str'  , skip_header=1, usecols=[0, 1]                  )
    org_perm_res     = np.genfromtxt(op_str, delimiter=',', dtype='float', skip_header=1, usecols=[2, 3, 4]               )
    rot_perm_samples = np.genfromtxt(rp_str, delimiter=',', dtype='str'  , skip_header=1, usecols=[0, 1]                  )
    rot_perm_res     = np.genfromtxt(rp_str, delimiter=',', dtype='float', skip_header=1, usecols=[2, 3, 4]               )
    por_samples      = np.genfromtxt(p_str,  delimiter=',', dtype='str'  , skip_header=1, usecols=[0, 1]                  )
    por_res          = np.genfromtxt(p_str,  delimiter=',', dtype='float', skip_header=1, usecols=[2, 3, 4, 5, 6, 7, 8, 9])
    
    # print(por_samples)

    header = '# sample_row, sample_column, size_row,size_col, phi_org[-], phi_org_effective[-], phi_rot[-], phi_rot_effective[-], k11[m^2], k12[m^2], k21[m^2], k22[m^2], kI[m^2], kII[m^2], angle[°], mean_sim_time[s]\n'

    if os.path.isfile(output_file_name):
        os.remove(output_file_name)


    with open(output_file_name, 'w') as f:
        # write header 
        f.write(header)
        for i in range(por_samples.shape[0]):
            key_row = str(por_samples[i, 0])
            key_col = str(por_samples[i, 1])

            row_number_org = -999
            for ii in range(org_perm_samples.shape[0]):
                if org_perm_samples[ii, 0] == key_row and org_perm_samples[ii, 1] == key_col:
                    row_number_org = ii

            row_number_rot = -999
            for ii in range(rot_perm_samples.shape[0]):
                if rot_perm_samples[ii, 0] == key_row and rot_perm_samples[ii, 1] == key_col:
                    row_number_rot = ii

            if not row_number_org == -999:
                k11 = org_perm_res[row_number_org, 0]
                k12 = org_perm_res[row_number_org, 1]
                ost = org_perm_res[row_number_org, 2]
            else:
                k11 = 0.0
                k12 = 0.0
                ost = 0.0

            if not row_number_rot == -999:
                k22 = rot_perm_res[row_number_rot, 0]
                k21 = rot_perm_res[row_number_rot, 1]
                rst = rot_perm_res[row_number_rot, 2]
            else:
                k22 = 0.0
                k21 = 0.0
                rst = 0.0

            kii = np.asarray( [ [k11, k12], [k21, k22] ] )

            _K  = ld.wrap_math.eigenvalues2d( kii )
            
            kI  = _K[0]
            kII = _K[1]

            if ( kii[0, 0] == 0.0 and kii[0, 1] == 0.0 ) or ( kii[1, 0] == 0.0 and kii[1, 1] == 0.0 ):
                angle = 0.0
            else:
                angle = ld.wrap_math.eigendir_angle_2d( kii )
            
            if ost == 0.0 and rst != 0.0:
                mean_sim_time = rst 
            elif ost != 0.0 and rst == 0.0:
                mean_sim_time = ost 
            elif ost == 0.0 and rst == 0.0:
                mean_sim_time = -999
            else:
                mean_sim_time = 0.5 * ( rst + ost )

            wline = f'{key_row},{key_col},{int(por_res[i,0])},{int(por_res[i,1])},{por_res[i,4]},{por_res[i,5]},{por_res[i,6]},{por_res[i,7]},{k11},{k12},{k21},{k22},{kI:.6g},{kII:.6g},{angle:1.3f},{mean_sim_time:1.2f}\n'
            f.write(wline)





def all_porosities(filelist, outfolder):
    """
    
    """

    # fmt        = '%s', '%1.5f'
    header = 'file, phi_org [-]\n'

    f = open(f'{outfolder}/porosities.csv', 'w')
    f.write(header)

    for myfilestr in filelist:
        array, size_ = ld.wrap_import.read_pgm(myfilestr)
        myfilestr = myfilestr.replace('.pgm', '')
        myfilestr = myfilestr.replace(f'{outfolder}/', '')
        porosity  = ld.porespace.porosity(array, zero_is_solid = False)    

        f.write(f'{myfilestr}, {porosity}\n')

    f.close()