import os
import argparse 

import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DP Finetuning')
    parser.add_argument(
        '--dataset', 
        default='cifar10', 
        type=str,                
        choices=[
            'celeba',
            'cifar10',
            'cifar100',
            'emnist',
            'fashionmnist',
            'mnist',
            'stl10',
            'svhn']
    )
    parser.add_argument(
        '--arch', 
        default="beitv2_large_patch16_224_in22k", 
        type=str
    )
    parser.add_argument(
        '--num-classes', 
        default=10, 
        type=int
    )
    parser.add_argument(
        '--lr', 
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--epochs', 
        default=10, 
        type=int
    )
    parser.add_argument(
        '--batch_size', 
        default=10000, 
        type=int
    )
    parser.add_argument(
        '--sigma', 
        default=10.0, 
        type=float
    )
    parser.add_argument(
        '--max_per_sample_grad_norm', 
        default=1, 
        type=int
    )
    parser.add_argument(
        "--disable_dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure_rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommender to turn it on.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1,
        metavar="E",
        help="Target epsilon (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11297,
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/scratch/gpfs/ashwinee/datasets/",
    )
    parser.add_argument(
        "--augmult",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
    return args

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def get_features(f, images, batch=256):
    features = []
    for img in tqdm(images.split(batch)):
        with torch.no_grad():
            img = F.interpolate(img.cuda(), size=(224, 224), mode="bicubic") # up-res size hardcoded
            features.append(f(img).detach().cpu())
    return torch.cat(features)

def download_things(args):
    dataset_path = args.dataset_path + args.dataset
    ds = getattr(datasets, args.dataset.upper())(dataset_path, transform=transforms.ToTensor(), train=True, download=True)
    feature_extractor = nn.DataParallel(timm.create_model(args.arch, num_classes=0, pretrained=True))

def extract_features(args, images_train, images_test):
    ### SET SEEDS
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ### GET DATA
    dataset_path = args.dataset_path + args.dataset

    ### GET PATH
    abbrev_arch = args.arch.split("_")[0]
    extracted_path = args.dataset_path + "transfer/features/" + args.dataset + "_" + abbrev_arch
    extracted_train_path = extracted_path + "_train.npy"
    extracted_test_path = extracted_path + "_test.npy"

    ### DO EXTRACTION

    print("GENERATING AND SAVING EXTRACTED FEATURES AT ", extracted_train_path)
    feature_extractor = nn.DataParallel(timm.create_model(args.arch, num_classes=0, pretrained=True)).eval().cuda()
    features_train = get_features(feature_extractor, images_train)
    features_test = get_features(feature_extractor, images_test)
    os.makedirs(extracted_path, exist_ok=True)
    np.save(extracted_train_path, features_train)
    np.save(extracted_test_path, features_test)

if __name__ == "__main__":
    args = parse_args()
    download_things(args)