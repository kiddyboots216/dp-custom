# dp-custom
Custom DP code in PyTorch to generate experiments for various papers.

To quickstart with DP finetuning, first install the dependencies and then run `python utils.py` to download the dataset and model you want to use (this is necessary for compute nodes with no internet access, e.g. a SLURM cluster). Then just run `sbatch run_classifier.sh`. 

The current version of Opacus needs some modifications to support things that we want to do. This is accomplished via maintaining a soft fork of the files that we want changed in custom_opacus and then shifting them via rsync. The script will need the path of this module and your conda env to do this.

Dependencies (non-exhaustive):
  - pytorch
  - numpy
  - torchvision
  - opacus
  - timm
  - tqdm
  - ema-pytorch
  - wandb
