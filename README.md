# dp-custom
Custom DP code in PyTorch to generate experiments for various papers.

To quickstart with DP finetuning for the conference submission, run the following command:

```
python {script}.py\
    --dataset_path ${0}\
    --dataset ${1}\
    --lr ${2}\
    --epsilon ${3}\
    --epochs ${4}\
    --arch ${5}\
```

For conventional CV experiments (ImageNet, CIFAR10, CIFAR100, FashionMNIST, EMNIST, MNIST, STL10, SVHN) `{script} = finetune_classifier_dp`.

For OOD experiments in Wilds (waterbirds, fmow, domainnet, camelyon) `{script} = wilds_finetune_classifier_dp` after following directions in the cited papers to download and split the datasets accordingly.

For OOD experiments transferring from CIFAR10/CIFAR100 first add the `"--save_weights"` flag when running a conventional CV experiment, then use `{script} = test_transfer` and specify the transfer dataset with `--transfer_dataset ${6}`. Make sure to download CIFAR10C/CIFAR100C from the cited paper.

Dependencies (non-exhaustive):
  - pytorch
  - numpy
  - torchvision
  - opacus
  - timm
  - tqdm
  - wandb
