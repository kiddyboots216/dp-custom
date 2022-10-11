#!/bin/bash

#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1
#SBATCH -t 0-1:00 # time requested (D-HH:MM)
# print some info for context
pwd
hostname
date

echo starting job...

# replace these as necessary
LIB_PATH=/scratch/gpfs/$USER/dp-custom/custom_opacus
CONDA_PATH=/scratch/gpfs/$USER/envs/opacus

module purge 
module load anaconda3/2022.5
conda activate $CONDA_PATH
ulimit -n 50000

# we need to modify opacus a bit
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/optimizers/topk_influence_optimizer.py $CONDA_PATH/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/optimizers/__init__.py $CONDA_PATH/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/privacy_engine.py $CONDA_PATH/lib/python3.8/site-packages/opacus/
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/accountants/influence_accountant.py $CONDA_PATH/lib/python3.8/site-packages/opacus/accountants
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/accountants/__init__.py $CONDA_PATH/lib/python3.8/site-packages/opacus/accountants

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

python finetune_classifier_dp.py