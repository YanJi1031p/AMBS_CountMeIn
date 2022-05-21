#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__  = '2020_08_01'

# This script loads the required modules for the preprocessing of IFS HRES data in scope of the
# downscaling application in scope of the MAELSTROM project on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements_preprocess.txt).

SCR_NAME_MOD="modules_train.sh"
HOST_NAME=`hostname`

# start loading modules
echo "%${SCR_NAME_MOD}: Start loading modules on ${HOST_NAME} required for CountMeIn challenge."

ml purge
ml use $OTHERSTAGES

ml Stages/2020
ml GCC/10.3.0
ml GCCcore/.10.3.0
ml ParaStationMPI/5.4.9-1
ml TensorFlow/2.5.0-Python-3.8.5

