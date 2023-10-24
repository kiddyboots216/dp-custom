#!/bin/bash

#SBATCH     --nodes=1               # node count
#SBATCH     --ntasks-per-node=1      # total number of tasks per node
#SBATCH     --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH     --mem=768G                # total memory per node (4 GB per cpu-core is default)
#SBATCH     --gres=gpu:4             # number of gpus per node
#SBATCH     --time=00:59:00          # total run time limit (HH:MM:SS)
##SBATCH    --partition=mig            # partition (queue)
#SBATCH    --constraint=gpu80         # constraint (e.g. gpu80)
#SBATCH     -o Report/%j.out            # STDOUT
##SBATCH     --mail-type=ALL          # send email on job start, end and fail
##SBATCH     --mail-user=ashwinee@princeton.edu      # email address to send to

# print some info for context
pwd
hostname
date

echo starting job...

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export N_WORKERS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))

python finetune_classifier_dp.py\
    --dataset_path "/scratch/gpfs/ashwinee/datasets/"\
    --dataset ImageNet\
    --arch vit_base_patch16 \
    --workers 8\
    --lr ${1}\
    --epsilon ${2}\
    --epochs ${3}\
    --batch_size ${4}\
    --optimizer ${5}\
    --sched ${6}\
    --max_phys_bsz 160146 \
    --feature_norm ${7}\
    --feature_mod ${8}\
    --sigma ${9}\

# --max_phys_bsz 160146\