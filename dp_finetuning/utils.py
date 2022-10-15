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
DATASET_TO_CLASSES = {
            'CelebA': -1,
            'CIFAR10': 10,
            'CIFAR100': 100,
            'EMNIST': 62,
            'FashionMNIST': 10,
            'MNIST': 10,
            'STL10': 10,
            'SVHN': 10}
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DP Finetuning')
    parser.add_argument(
        '--dataset', 
        default='CIFAR10', 
        type=str,                
        choices=list(DATASET_TO_CLASSES.keys())
    )
    parser.add_argument(
        '--arch', 
        default="beitv2_large_patch16_224_in22k", 
        type=str
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
        default=-1, 
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    args.num_classes = DATASET_TO_CLASSES[args.dataset]
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

    ds = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), train=True, download=True)
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
    extracted_path = args.dataset_path + "transfer/features/" + args.dataset.lower() + "_" + abbrev_arch
    extracted_train_path = extracted_path + "/_train.npy"
    extracted_test_path = extracted_path + "/_test.npy"

    ### DO EXTRACTION

    print("GENERATING AND SAVING EXTRACTED FEATURES AT ", extracted_train_path)
    feature_extractor = nn.DataParallel(timm.create_model(args.arch, num_classes=0, pretrained=True)).eval().cuda()
    features_train = get_features(feature_extractor, images_train)
    features_test = get_features(feature_extractor, images_test)
    os.makedirs(extracted_path, exist_ok=True)
    np.save(extracted_train_path, features_train)
    np.save(extracted_test_path, features_test)

def get_ds(args):
    dataset_path = args.dataset_path
    abbrev_arch = args.arch.split("_")[0]
    extracted_path = args.dataset_path + "transfer/features/" + args.dataset.lower() + "_" + abbrev_arch
    extracted_train_path = extracted_path + "/_train.npy"
    extracted_test_path = extracted_path + "/_test.npy"

    if not os.path.exists(dataset_path):
        raise Exception('We cannot download a dataset/model here \n Run python utils.py to download things')
    if args.dataset in ["SVHN", "STL10"]:
        ds = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), split='train')
        images_train, labels_train = torch.tensor(ds.data) / 255.0, torch.tensor(ds.labels)
        ds = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), split='test')
        images_test, labels_test = torch.tensor(ds.data) / 255.0, torch.tensor(ds.labels)
    elif "MNIST" in args.dataset:
        if args.dataset == "EMNIST":
            ds_train = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), split='byclass', train=True)
            ds_test = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), split='byclass', train=False)
        else:
            ds_train = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), train=True)
            ds_test = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), train=False)
        images_train, labels_train = torch.tensor(ds_train.data.unsqueeze(1).repeat(1, 3, 1, 1)).float() / 255.0, torch.tensor(ds_train.targets)
        images_test, labels_test = torch.tensor(ds_test.data.unsqueeze(1).repeat(1, 3, 1, 1)).float() / 255.0, torch.tensor(ds_test.targets)
    else:
        ds = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), train=True)
        images_train, labels_train = torch.tensor(ds.data.transpose(0, 3, 1, 2)) / 255.0, torch.tensor(ds.targets)
        ds = getattr(datasets, args.dataset)(dataset_path, transform=transforms.ToTensor(), train=False)
        images_test, labels_test = torch.tensor(ds.data.transpose(0, 3, 1, 2)) / 255.0, torch.tensor(ds.targets)
    len_test = labels_test.shape[0]
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    if not os.path.exists(extracted_path):
        extract_features(args, images_train, images_test)
    if args.augmult > 0:
        ds_train = make_finetune_augmult_dataset(args, images_train, labels_train)
        if args.batch_size == -1:
            args.batch_size = len(ds_train)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_func)
    else:
        x_train = np.load(extracted_train_path)
        features_train = torch.from_numpy(x_train)
        ds_train = dataset_with_indices(TensorDataset)(features_train, labels_train)
        if args.batch_size == -1:
            args.batch_size = len(ds_train)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    x_test = np.load(extracted_test_path)
    features_test = torch.from_numpy(x_test)
    ds_test = dataset_with_indices(TensorDataset)(features_test, labels_test)
    test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, **kwargs)
    return train_loader, test_loader, features_test.shape[-1], len(labels_test)

from typing import Any, Callable, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torchvision

import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 

def my_collate_func(batch):
    batch = default_collate(batch)
    # import pdb; pdb.set_trace()
    #assert(len(batch) == 2)
    # batch_size, num_aug, channels, height, width = batch[0].size()
    bsz, num_aug, num_features = batch[0].size()
    batch[0] = batch[0].view([bsz * num_aug, num_features])
    batch[1] = batch[1].view([bsz * num_aug])
    return batch

class Augmult:
    def __init__(self, 
                image_size: Sequence[int],
                augmult: int,
                random_flip: bool,
                random_crop: bool,
                crop_size: Optional[Sequence[int]] = None,
                pad: Optional[int] = None,
    ):
        """
        image_size: new size for the image.
        augmult: number of augmentation multiplicities to use. This number should
        be non-negative (this function will fail if it is not).
        random_flip: whether to use random horizontal flips for data augmentation.
        random_crop: whether to use random crops for data augmentation.
        crop_size: size of the crop for random crops.
        pad: optional padding before the image is cropped.
        """
        self.augmult = augmult
        self.image_size = image_size
        self.random_flip = random_flip
        self.random_crop = random_crop
        # initialize some torchvision transforms
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.random_crop = transforms.RandomCrop(
                    size = (crop_size if crop_size is not None else image_size),
                )
        self.pad = pad
        if self.pad:
            self.padding = transforms.Pad(pad, padding_mode='reflect')
        
    def apply_augmult(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements data augmentation (Hoffer et al., 2019; Fort et al., 2021)
        
        Args:
            image: (single) image to augment.
            label: label corresponding to the image (not modified by this function).
        Returns:
            images: augmented images with a new prepended dimension of size `augmult`.
            labels: repeated labels with a new prepended dimension of size `augmult`.
        """
        image = torch.reshape(image, self.image_size)
        
        if self.augmult == 0:
            # we need to add a new dim bc the resulting code will expect it
            images = torch.unsqueeze(image, 0)
            labels = np.expand_dims(label, 0)
        elif self.augmult > 0:
            raw_image = torch.clone(image)
            augmented_images = []
            
            for _ in range(self.augmult):
                image_now = raw_image
                
                if self.random_crop:
                    if self.pad:
                        image_now = self.padding(image_now)
                    image_now = self.random_crop(image_now)
                if self.random_flip:
                    image_now = self.random_horizontal_flip(image_now)
                augmented_images.append(image_now)
            images = torch.stack(augmented_images, 0)
            labels = np.stack([label] * self.augmult, 0)
        else:
            raise ValueError('Augmult should be non-negative.')
        
        return images, labels

def make_finetune_augmult_dataset(args, images_train, labels_train):
    normalize = transforms.Normalize(mean=[x for x in [125.3, 123.0, 113.9]],
                                     std=[x for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            normalize,
            ])
    ds = FinetuneAugmultDataset(
        arch = args.arch,
        data = images_train,
        labels = labels_train,
        transform=transform_train,
        image_size=(3, 32, 32),
        augmult=16,
        random_flip=True,
        random_crop=True,
        crop_size=32,
        pad=4,
    )
    return ds

class FinetuneAugmultDataset(torch.utils.data.Dataset):
    """ Dataset Class Wrapper """

    def __init__(self, 
                arch: str,
                data: torch.Tensor,
                labels: torch.Tensor,
                image_size: Sequence[int],
                augmult: int,
                random_flip: bool,
                random_crop: bool,
                crop_size: Optional[Sequence[int]] = None,
                pad: Optional[int] = None,
                train: bool = True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                download: bool=False,
        ) -> None:
        self.data = data
        self.targets = labels
        self.augmult_module = Augmult(
            image_size = image_size,
            augmult = augmult,
            random_flip = random_flip,
            random_crop = random_crop,
            crop_size = crop_size,
            pad = pad,
        )
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.download = download
        self.feature_extractor = nn.DataParallel(timm.create_model(arch, num_classes=0, pretrained=True)).eval().cuda()

    def __getitem__(self, index: int) -> Tuple[Any,Any,Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        img, target = self.augmult_module.apply_augmult(img, target)
        with torch.no_grad():
            features = self.feature_extractor(F.interpolate(img.cuda(), size=(224, 224), mode="bicubic")).detach()
        return features, target, index

    def __len__(self):
        return len(self.targets)

if __name__ == "__main__":
    from easydict import EasyDict
    args = parse_args()
    import pdb
    import code

    class MyPdb(pdb.Pdb):
        def do_interact(self, arg):
            code.interact("*interactive*", local=self.curframe_locals)

    ds = get_ds(args)
    train_dataloader = DataLoader(ds,collate_fn=my_collate_func, batch_size=12, shuffle=True)
    data, label, indices = next(iter(train_dataloader))
