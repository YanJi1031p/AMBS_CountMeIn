#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=hai_countmein

ml purge

ml Stages/2020  GCC/9.3.0  ParaStationMPI/5.4.7-1 OpenCV/4.5.0-Python-3.8.5
ml GDAL/3.1.2-Python-3.8.5
ml scikit 

source /p/project/atmlaml/kamath2/helmholtz-ai-conference-challenge/venv/bin/activate

# srun python train.py 
srun python starter-pack.py --data_path_So2Sat_pop_part1 '/p/project/hai_countmein/data/So2Sat_POP_Part1' --data_path_So2Sat_pop_part2 '/p/project/hai_countmein/data/So2Sat_POP_Part2'
