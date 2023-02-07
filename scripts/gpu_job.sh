#! /bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=wally
#SBATCH --gres=gpu

echo "ACTIVATE BASH"
source /users/${USER}/.bashrc
source activate /scratch/users/${USER}/conda/wally

echo "\nMODULES"
module load cuda/11.4.2-gcc-9.4.0
module load cudnn/8.2.4.15-11.4-gcc-9.4.0
module list

echo "\nRUNNING EXPERIMENT"
cd /users/k21158663/where-is-wally-
python train.py