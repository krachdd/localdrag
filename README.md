# Preprocessing Code for adapted Drag terms in pseudo-3D Stokes Simulations
[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--4313-d45815.svg)](https://doi.org/10.18419/darus-4313)
[![Identifier](https://img.shields.io/badge/Publication-blue)]([http://ssrn.com/abstract=4927521](https://doi.org/10.1016/j.advwatres.2024.104860))

Create geometry informed pre-factor maps for pseudo-3D Stokes simulations with Dumux based on local pore morphology. 
The README is kept short. Please check the comments in the source code and the details in the paper for more information.

## How to
Clone the repository and use 
```python
import sys
sys.path.append('PATH_TO_SRC')

import localdrag as ld
from localdrag import *
```
to make the interpreter search in for the required module. Required packages are listed in `requirements.txt`. Modifying the `PYTHONPATH` environment variable is also possible.


## Create geometry informed drag terms

DumuX requires three files to employ an geometry informed drag term: $\lambda_1$, $\lambda_2$, $h(\mathbf{x})$. Write these by using the modules functions.

### From 3D raw data
See folder : `test/2d_single_precipitate`

```python
h_map_scaled, lambda1, lambda2 = ld.create_lambda_maps.lambda_total_map_3d(
                                                                            geom, 
                                                                            voxelsize, 
                                                                            channelwidth, 
                                                                            solidframe, 
                                                                            crosssection, 
                                                                            smooth, 
                                                                            sigma, 
                                                                            cs_weight
                                                                          )

# Write the domains for Dumux
ld.write_maps.write2txt(outpath, fn, 'lambda1', lambda1)
ld.write_maps.write2txt(outpath, fn, 'lambda2', lambda2)
ld.write_maps.write2pgm(outpath, fn, 'hx', h_map_scaled) # may be same as input
```

### From 2D image data
See folder : `test/2d_single_precipitate`

```python
lambda1, lambda2 = ld.create_lambda_maps.lambda_total_map(
                                                            h_map01, 
                                                            voxelsize, 
                                                            vox_per_height, 
                                                            channelwidth, 
                                                            solidframe, 
                                                            smooth, 
                                                            sigma, 
                                                            crosssection, 
                                                            cs_weight
                                                          )

# Write the domains for Dumux
ld.write_maps.write2txt(outpath, fn, 'lambda1', lambda1)
ld.write_maps.write2txt(outpath, fn, 'lambda2', lambda2)
ld.write_maps.write2pgm(outpath, fn, 'hx', h_map_scaled)
```


### Convert 3D data to 2D 

Use `hmap_from_3d.py` to create a `hmap*.pgm` file from 3D geometry. For all further details see section *From 2D image data*.


## Parameters
```bash
# Relevant parameters 
"""
h_map01 : numpy ndarray
    2d, relative height of geometry. Scaled
    between 0 (fully blocked/solid voxel) and 
    1 (fully fluid voxel).

voxelsize : float [ m ]
    Voxelsize of the domain.

vox_per_height : int 
    Number of voxels per h_Omega.

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

lambda1 : numpy ndarray
    2d map containing factors to use in DUMUX simulator
    in main direction (z, 2nd dir of 2d array, main pressure gradient)
lambda2 : numpy ndarray
    2d map containing factors to use in DUMUX simulator
    in perpendicular direction 


geom : numpy.ndarray
    3d domain of the porous material.
    Zero -> indicates fluid voxel.
    One  -> indictaes solid voxel.

h_map_scaled :numpy ndarray
    2d, height of geometry
    scaled by the voxelsize

height : float
    h_Omega of the domain  [ m ]


smooth : bool 
    Smooth the gradient before factor computation
sigma : float
    Standard deviation for Gaussian kernel.

cmap : numpy.ndarray 
    Colum-wise check, w in this voxel. 
rmap : numpy.ndarray
    Row-wise check, l  in this voxel.
"""
```

## License

The solver is licensed under the terms and conditions of the MIT License (MIT) version 3 or - at your option - any later
version. The License can be [found online](https://opensource.org/license/mit/) or in the LICENSE.md file
provided in the topmost directory of source code tree.

## How to cite

The solver is research software and developed at a research institute. Please cite **specific releases** according to [**DaRUS**](https://doi.org/10.18419/darus-4313) version.

If you are using localdrag in scientific publications and in the academic context, please cite our publications:

```bib
@article{Krach2025,
    title = {A novel geometry-informed drag term formulation for pseudo-3D Stokes simulations with varying apertures},
    journal = {Advances in Water Resources},
    volume = {195},
    year = {2025},
    doi = {https://doi.org/10.1016/j.advwatres.2024.104860},
    author = {David Krach and Felix Weinhardt and Mingfeng Wang and Martin Schneider and Holger Class and Holger Steeb},
    keywords = {Porous media, Stokes flow, Biomineralization, Microfluidics, Image-based simulations, Computational efficiency versus accuracy}
}
```

```bib
@data{Krach2024b,
    author = {Krach, David and Weinhardt, Felix and Wang, Mingfeng and Schneider, Martin and Class, Holger and Steeb, Holger},
    publisher = {DaRUS},
    title = {{A novel geometry-informed drag term formulation for pseudo-3D Stokes simulations with varying apertures}},
    year = {2024},
    version = {DRAFT VERSION},
    doi = {10.18419/darus-4313},
    url = {https://doi.org/10.18419/darus-4313}
}
```


## Links to Stokes Solvers

Links to the different numerical solver employed in the paper are listed below.

1. poremaps: [**DaRUS**](https://doi.org/10.18419/darus-3676), [**GIT**](https://git.rwth-aachen.de/david.krach/poremaps)
2. Dumux : [**DaRUS**](https://doi.org/10.18419/darus-XYXY), [**GIT**](https://git.iws.uni-stuttgart.de/dumux-repositories/dumux)
3. OpenFOAM : [**GIT**](https://develop.openfoam.com/Development/openfoam)

## Acknowledgements
Funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy (Project number 390740016 - EXC 2075 and the Collaborative Research Center 1313 (project number 327154368 - SFB1313). We acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

## Developer

- [David Krach](https://www.mib.uni-stuttgart.de/institute/team/Krach/) E-mail: [david.krach@mib.uni-stuttgart.de](mailto:david.krach@mib.uni-stuttgart.de)
- [Felix Weinhardt](https://www.mib.uni-stuttgart.de/de/institut/team/Weinhardt-00003/) E-mail: [felix.weinhardt@mib.uni-stuttgart.de](mailto:felix.weinhardt@mib.uni-stuttgart.de)

## Contact
- [Software Support Institute of Applied Mechanics](mailto:software@mib.uni-stuttgart.de)
- [Data Support Institute of Applied Mechanics](mailto:data@mib.uni-stuttgart.de)
