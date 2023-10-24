#!/bin/bash

lrs=(10)
epsilons=(7.9)
epochs=(250)
bszs=(-1)
# bszs=(320292 640584)
optimizers=("sgd")
sched=(0)
feature_norm=(50)
feature_mod=("avgpool")

# Loop over the learning rates
for lr in "${lrs[@]}"; do
  # Loop over the epochs values
  for epoch in "${epochs[@]}"; do
    # Loop over the epsilon values
    for epsilon in "${epsilons[@]}"; do
      # Loop over the batch sizes
      for bsz in "${bszs[@]}"; do
        # Loop over the optimizers
        for opt in "${optimizers[@]}"; do
          # Loop over the weight decay values
          for sch in "${sched[@]}"; do
            # Loop over the feature norms
            for fn in "${feature_norm[@]}"; do
              # Loop over the feature mods
              for fm in "${feature_mod[@]}"; do
                # Print the current configuration
                echo "Submitting with lr=$lr, epsilon=$epsilon, epochs=$epoch, batch_size=$bsz, optimizer=$opt, sched=$sch, fn=$fn, fm=$fm"
                # Submit the job
                sbatch imagenet.sh $lr $epsilon $epoch $bsz $opt $sch $fn $fm
              done
            done
          done
        done
      done
    done
  done
done
