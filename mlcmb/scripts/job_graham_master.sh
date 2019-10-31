#!/bin/bash
#SBATCH --account=moritzm
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
#SBATCH --time=0-00:30            # time (DD-HH:MM)

module load python
module load scipy-stack
source ~/ENV/bin/activate

python ~/mlcmb/mlcmb/trainingdata.py  ~/mlcmb/mlcmb/configs/config_graham_1.ini