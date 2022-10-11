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
    batch_size, num_aug, channels, height, width = batch[0].size()
    batch[0] = batch[0].view([batch_size * num_aug, channels, height, width])
    batch[1] = batch[1].view([batch_size * num_aug])
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

class DatasetLoader(torchvision.datasets.CIFAR10):
    """ Dataset Class Wrapper """

    def __init__(self, 
                root: str,
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
        super().__init__(root, train, transform, target_transform, download)
        self.augmult_module = Augmult(
            image_size = image_size,
            augmult = augmult,
            random_flip = random_flip,
            random_crop = random_crop,
            crop_size = crop_size,
            pad = pad,
        )

    def __getitem__(self, index: int) -> Tuple[Any,Any,Any]:

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        img, target = self.augmult_module.apply_augmult(img, target)
        return img, target, index

if __name__ == "__main__":
    import torchvision
    from augmult import Augmult
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    train_dataset = DatasetLoader(
            root='../data', 
            train=True, 
            download=True,
            transform=transform_train,
            image_size=(3, 32, 32),
            augmult=16,
            random_flip=True,
            random_crop=True,
            crop_size=32,
            pad=4,
        )

    # test_collate_fn = lambda batch: default_collate(batch).reshape(batch.shape[:2].numel(), batch.shape[2], batch.shape[3], batch.shape[4])
    train_dataloader = DataLoader(train_dataset,collate_fn=my_collate_func, batch_size=12, shuffle=True)
    data, label, indices = next(iter(train_dataloader))
    import pdb; pdb.set_trace()

    