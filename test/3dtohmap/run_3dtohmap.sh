#!/bin/sh
SOURCEDIR="$(dirname $(dirname "$(pwd)"))"
export PYTHONPATH=$PYTHONPATH:${SOURCEDIR}

python3 hmap_from_3d.py -dir ../rawfiles -vs 1e-6 -v_height 36.0
