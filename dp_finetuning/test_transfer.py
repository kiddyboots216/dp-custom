import os

import timm
import torch
import opacus
import numpy as np
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import wrap_data_loader

from utils import parse_args, extract_features, get_ds, DATASET_TO_CLASSES, ARCH_TO_NUM_FEATURES

import pdb
import code

args = None
len_test = None

def load_grads_weights(args):
    """
    Load noisy grads, raw grads and weights from the checkpoint path corresponding to args
    """
    noisy_grads, raw_grads, weights = None, None, None
    f = f"grad_datasets/{args.arch}/{args.dataset}/grads_weights_{args.num_runs}_{args.epochs}_{int(args.lr)}_{int(args.epsilon)}.npz"
    print(f"Loading grads and weights from {f}")
    loaded_arrays = np.load(f, allow_pickle=True)
    noisy_grads = loaded_arrays["noisy_grads"]
    raw_grads = loaded_arrays["raw_grads"]
    weights = loaded_arrays["weights"]
    return noisy_grads, raw_grads, weights

def load_weights(args):
    """
    Load weights from the checkpoint path corresponding to args
    """
    weights = None
    f = f"ckpts/{args.arch}/{args.dataset}/weights_{args.num_runs}_{args.epochs}_{int(args.lr)}_{int(args.epsilon)}.npz"
    print(f"Loading weights from {f}")
    loaded_arrays = np.load(f, allow_pickle=True)
    weights = loaded_arrays["weights"]
    return weights

def set_weights(model, args):
    """
    load the weights from the checkpoint path corresponding to args
    set the weights to the model
    """
    # noisy_grads, raw_grads, weights = load_grads_weights(args)
    final_weights = load_weights(args)
    final_weights = torch.from_numpy(final_weights)
    # final_weights = torch.from_numpy(weights[-1,-1])
    final_weights = final_weights.to(args.device).view_as(model.weight.data)
    model.weight.data.zero_()
    model.weight.data.add_(final_weights)
    return model

def test(args, model, test_loader):
    device = args.device
    criterion = nn.CrossEntropyLoss()
    acc = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            corr = pred.eq(target.view_as(pred))
            acc += corr.sum().item() * 100/target.shape[0]
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def main():
    global args
    global len_test
    args = parse_args()
    num_features = ARCH_TO_NUM_FEATURES[args.arch]
    model = nn.Linear(num_features, args.num_classes, bias=False).cuda()
    model = set_weights(model, args)
    # hardcode the model to eval mode
    model.eval()
    # set the new dataset for transfer learning
    args.dataset = args.transfer_dataset
    _, test_loader, _, _ = get_ds(args) # get the new dataset
    test(args, model, test_loader)
    
if __name__ == '__main__':
    main()
    