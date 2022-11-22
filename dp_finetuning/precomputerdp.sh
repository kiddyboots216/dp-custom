#!/bin/bash

#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:0
#SBATCH -t 0-01:00 # time requested (D-HH:MM)
#SBATCH -o Report/%j.out # STDOUT
#SBATCH --array=30-95:5

# print some info for context
pwd
hostname
date

echo starting job...

# replace these as necessary
LIB_PATH=/home/$USER/dp-custom/custom_opacus
CONDA_PATH=/data/nvme/$USER/anaconda3/envs/opacus
source ~/.bashrc
conda activate $CONDA_PATH
ulimit -n 50000

# we need to modify opacus a bit
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/optimizers/topk_influence_optimizer.py $CONDA_PATH/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/optimizers/optimizer.py $CONDA_PATH/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/optimizers/__init__.py $CONDA_PATH/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/privacy_engine.py $CONDA_PATH/lib/python3.8/site-packages/opacus/
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/accountants/influence_accountant.py $CONDA_PATH/lib/python3.8/site-packages/opacus/accountants
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" $LIB_PATH/accountants/__init__.py $CONDA_PATH/lib/python3.8/site-packages/opacus/accountants

srun python pre_compute_rdp.py --sigma ${SLURM_ARRAY_TASK_ID}