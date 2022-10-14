#!/bin/bash

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

python finetune_classifier_dp.py\
    --dataset ${1}\
    --lr ${2}\
    --max_per_sample_grad_norm ${3}\
    --epsilon ${4}\
    --sigma ${5}\
    --epochs ${6}\
    --batch_size ${7}\
    ${8}\