#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__  = '2020_08_01'

# This script loads the required modules for the preprocessing of IFS HRES data in scope of the
# downscaling application in scope of the MAELSTROM project on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements_preprocess.txt).

SCR_NAME_MOD="modules_preprocess.sh"
HOST_NAME=`hostname`

# start loading modules
echo "%${SCR_NAME_MOD}: Start loading modules on ${HOST_NAME} required for CountMeIn challenge."

ml purge
ml use $OTHERSTAGES

ml Stages/2020  GCC/9.3.0  ParaStationMPI/5.4.7-1 OpenCV/4.5.0-Python-3.8.5
ml GDAL/3.1.2-Python-3.8.5
ml scikit 


