#!/bin/sh
SOURCEDIR="$(dirname $(dirname "$(pwd)"))"
export PYTHONPATH=$PYTHONPATH:${SOURCEDIR}

python3 transform3dto2d.py -dir test_3d_single_precipitate -vs 1e-6 -height 36.0e-6

