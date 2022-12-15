# dp-custom
Custom DP code in PyTorch to generate experiments for various papers.

To quickstart with DP finetuning, just run the code in the top-level .ipynb. For this, you don't need to install any other code in the library, so feel free to just copy paste this into your own Notebook or Colab. Implementations of weight averaging, data augmentation, and the free step can be found in `dp_finetuning'.

Dependencies (non-exhaustive):
  - pytorch
  - numpy
  - torchvision
  - opacus
  - timm
  - tqdm
  - wandb
