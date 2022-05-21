#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gres gpu:4
#SBATCH --partition accelerated
#SBATCH --exclusive

ml --force  purge

# srun python train.py 
srun singularity run --bind /hkfs /hkfs/home/dataset/datasets/So2Sat_POP/countmein_sklearn_1.0.sif  python3 starter-pack.py --data_path_So2Sat_pop_part1 '/hkfs/home/dataset/datasets/So2Sat_POP/So2Sat_POP_Part1' --data_path_So2Sat_pop_part2 '/hkfs/home/dataset/datasets/So2Sat_POP/So2Sat_POP_Part2'
