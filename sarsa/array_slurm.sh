#!/bin/bash

#SBATCH --array=0-599
#SBATCH -c 10
#SBATCH --mem=8g
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o logs/slurm/slurm-%A_%a.out


# mila-cluster specific module:
module load pytorch

python main_sarsa.py --array_name=Jun6_10M --array_id=$SLURM_ARRAY_TASK_ID 
