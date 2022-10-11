#!/bin/bash

#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
##SBATCH --constraint=gpu80
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:1
#SBATCH -t 0-12:00 # time requested (D-HH:MM)
# print some info for context
pwd
hostname
date

echo starting job...

# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
# source ~/.bashrc
# conda init
module purge 
module load anaconda3/2022.5
conda activate /scratch/gpfs/ashwinee/envs/opacus/
ulimit -n 50000

# do ALL the research
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/optimizers/optimizer.py envs/opacus/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/optimizers/topk_optimizer.py envs/opacus/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/optimizers/topk_influence_optimizer.py envs/opacus/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/optimizers/utils.py envs/opacus/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/optimizers/topk_utils.py envs/opacus/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/optimizers/__init__.py envs/opacus/lib/python3.8/site-packages/opacus/optimizers
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/privacy_engine.py envs/opacus/lib/python3.8/site-packages/opacus/
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/accountants/influence_accountant.py envs/opacus/lib/python3.8/site-packages/opacus/accountants
rsync -zarh --exclude ".git/*" --exclude "*.pyc" --exclude "*.out" custom_opacus/accountants/__init__.py envs/opacus/lib/python3.8/site-packages/opacus/accountants

# cd /data/nvme/ashwinee/SparsefluenceDP
# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=TRUE
export WANDB_MODE=offline
python train.py\
    --dataset ${1}\
    --epsilon ${2}\
    --max_per_sample_grad_norm ${3}\
    --batch_size ${4}\
    --lr ${5}\
    --sigma ${6}\
    --epochs ${7}\
    --model ${8}\
    ${9}\
    ${10}\
    ${11}\