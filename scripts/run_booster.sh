#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=hai_countmein
#SBATCH --output=../logs/run_training-out.%j
#SBATCH --error=../logs/run_training-err.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.ji@fz-juelich.de
# hai_countmein; deepacf

srun python training_swint_cla_rf_reg.py --model_folder '/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_model/swintL_cate7_rf_reg'
# srun python predicting_rf_reg.py --model_folder '/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_model/cate_rf_reg'
# srun python training_rf_reg.py --model_folder '/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_model/cate7_rf_reg_fine_tuing' 
# srun python training_swint.py --train_start_idx_list [6,8,5,9,2] --seed 0 --dropout_rate 0.02 --embed_dim 64 --num_mlp 256 --num_epochs 40 --save_dir '/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_model/epoch40swinL' --norm 'MaxMin' --num_heads 12
