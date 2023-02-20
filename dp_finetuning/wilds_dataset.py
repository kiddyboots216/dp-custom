from wilds import get_dataset

from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np

import os
from PIL import Image


def get_indices_by_value(values):
    # Returns list_of_lists, where list_of_lists[i] corresponds to all indices
    # i with equal values[i].
    index_dict = defaultdict(list)
    for i, v in zip(range(len(values)), values):
        index_dict[v].append(i)
    return list(index_dict.values())


def equal_subsample(list_of_lists, rng):
    min_size = np.min([len(l) for l in list_of_lists])
    subsampled = [rng.choice(l, size=min_size, replace=False) for l in list_of_lists]
    subsampled = np.concatenate(subsampled)
    return subsampled


class WILDS(Dataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        meta_selector=None,
        transform=None,
        download=True,
        return_meta=False,
        subsampled_y=False,
        subsampled_meta=False,
        seed=0,
    ):
        # Split can be train, id_val, id_test, val, test.
        super().__init__()
        parent_dataset_name = dataset_name
        if "waterbirds" in dataset_name:
            parent_dataset_name = "waterbirds"
        full_dataset = get_dataset(
            dataset=parent_dataset_name, download=download, root_dir=root
        )
        dataset = full_dataset.get_subset(split, transform=None)
        self._dataset_name = dataset_name
        self._transform = transform
        self._dataset = dataset
        self._indices = None
        self._meta_selector = None
        self._return_meta = return_meta
        self._rng = np.random.default_rng(seed=seed)
        if meta_selector is not None:
            self._meta_selector = tuple(meta_selector)
            super_indices = self._dataset.indices
            subset_metadata = self._dataset.dataset.metadata_array[
                super_indices
            ].numpy()
            mask = np.all(subset_metadata == np.array(meta_selector), axis=-1)
            # For some reason the indices is 2d  (each index is in its own list), so need [:. 0] below.
            self._indices = np.argwhere(mask)[:, 0]
        if subsampled_y or subsampled_meta:
            if meta_selector is not None or (subsampled_y and subsampled_meta):
                raise ValueError(
                    f"subsampled_y ({subsampled_y}), subsampled_meta ({subsampled_meta}), "
                    f"meta_selector ({meta_selector}) but two of these three must be None."
                )
            super_indices = self._dataset.indices
            if subsampled_y:
                subset_ys = self._dataset.dataset._y_array[super_indices].numpy()
                list_of_lists = get_indices_by_value(subset_ys)
            if subsampled_meta:
                subset_metas = self._dataset.dataset.metadata_array[
                    super_indices
                ].numpy()
                subset_metas = [tuple(l) for l in subset_metas]
                list_of_lists = get_indices_by_value(subset_metas)
            self._indices = equal_subsample(list_of_lists, self._rng)

    def __getitem__(self, i):
        if self._indices is None:
            x, y, z = self._dataset[i]
        else:
            x, y, z = self._dataset[self._indices[i]]
            if self._meta_selector is not None:
                assert tuple(z) == self._meta_selector
        if self._transform is not None:
            x = x.convert("RGB")
            x = self._transform(x)
        if "waterbirds" in self._dataset_name and "background" in self._dataset_name:
            y = z[0]
        if self._return_meta:
            return x, y, z
        return x, y

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._dataset)


class WILDSTensor(Dataset):
    def __init__(self, features, dataset, labels):
        super().__init__()
        self.features = features
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, i):
        return self.features[i], self.labels[i]

    def __len__(self):
        return len(self.dataset)


class WILDSFeatures(Dataset):
    def __init__(self, features, dataset):
        super().__init__()
        self.features = features
        self.dataset = dataset

    def __getitem__(self, i):
        # return self.features[i], self.labels[i]
        # i is in range(len(self.dataset)), that is shorter than the real dataset
        # we go to indices to map this to the indices of range real dataset
        x, y = self.dataset.__getitem__(i)
        if self.dataset._indices is None:
            x = self.features[i]
            return x, y
        x = self.features[self.dataset._indices[i]]
        return x, y

    def __len__(self):
        return len(self.dataset)

    def get_meta_selector(self):
        meta_selector = self.dataset._meta_selector
        if meta_selector == (0, 0, 0):
            return "landbg - landbird"
        elif meta_selector == (0, 1, 0):
            return "landbg - waterbird"
        elif meta_selector == (1, 0, 0):
            return "waterbg - landbird"
        elif meta_selector == (1, 1, 0):
            return "waterbg - waterbird"
        elif meta_selector is None:
            return "base"
        else:
            raise ValueError(f"Unknown meta_selector {meta_selector}")


class Fmow(Dataset):
    def __init__(self, root, regions, split="train", transform=None):
        # Split can be train, id_val, id_test, val, test.
        # regions is a lister of integers between 0 and 4 denoting the regions, don't use 'Other'.
        # For fmow root is the directory that contains fmow (and other wilds datasets).
        super().__init__()
        super_dataset = get_dataset(dataset="fmow", download=False, root_dir=root)
        self._subset = super_dataset.get_subset(split, transform=transform)
        self._regions = regions
        if "all" not in self._regions:
            super_indices = self._subset.indices
            self.super_indices = super_indices
            subset_metadata = self._subset.dataset.metadata_array[super_indices].numpy()
<<<<<<< Updated upstream
            self.subset_metadata = subset_metadata
            self._indices = np.argwhere([(a in regions) for a in subset_metadata[:, 0]])[:,0]
        
=======
            self._indices = np.argwhere(
                [(a in regions) for a in subset_metadata[:, 0]]
            )[:, 0]

>>>>>>> Stashed changes
    def __getitem__(self, i):
        if "all" not in self._regions:
            x, y, _ = self._subset[self._indices[i]]
        else:
            x, y, _ = self._subset[i]
        return x, y

    def __len__(self) -> int:
        if "all" not in self._regions:
            return len(self._indices)
        else:
            return len(self._subset)
        
class FmowTensor(Dataset):
    def __init__(self, features, dataset):
        super().__init__()
        self.features = features
        self.y_array = dataset._subset.y_array
        self._indices = dataset._indices
        self._regions = dataset._regions
    
    def __getitem__(self, i):
        x = self.features[i]
        y = self.y_array[self._indices[i]]
        return x, y
    
    def __len__(self) -> int:
        return len(self._indices)


VALID_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

SENTRY_DOMAINS = ["clipart", "painting", "real", "sketch"]

NUM_CLASSES_DICT = {"full": 345, "sentry": 40}

VALID_SPLITS = ["train", "test"]

VALID_VERSIONS = ["full", "sentry"]

<<<<<<< Updated upstream
ROOT = '/data/nvme/$USER/datasets/domainnet'
=======
ROOT = "/data/nvme/ashwinee/datasets/domainnet"
>>>>>>> Stashed changes
SENTRY_SPLITS_ROOT = ROOT


def load_dataset(root, domains, split, version):
    if len(domains) == 1 and domains[0] == "all":
        if version == "sentry":
            domains = SENTRY_DOMAINS
        else:
            domains = VALID_DOMAINS

    data = []
    for domain in domains:
        if version == "sentry":
            if os.path.isdir(root + "/SENTRY_splits"):
                idx_file = os.path.join(
                    root, f"SENTRY_splits/{domain}_{split}_mini.txt"
                )
            else:
                idx_file = os.path.join(
                    SENTRY_SPLITS_ROOT, f"{domain}_{split}_mini.txt"
                )
        else:
            if os.path.isfile(root + f"/{domain}_{split}.txt"):
                idx_file = os.path.join(root, f"{domain}_{split}.txt")
            else:
                idx_file = os.path.join(ROOT, f"{domain}_{split}.txt")
        with open(idx_file, "r") as f:
            data += [line.split() for line in f]
    return data


class DomainNet(Dataset):
    def __init__(
        self,
        domain,
        split="train",
        root=ROOT,
        transform=None,
        unlabeled=False,
        verbose=True,
        version="sentry",
    ):
        super().__init__()

        if version not in VALID_VERSIONS:
            raise ValueError(
                f"dataset version must be in {VALID_VERSIONS} but was {version}"
            )
        domain_list = domain.split(",")
        for domain in domain_list:
            if domain != "all" and version == "full" and domain not in VALID_DOMAINS:
                raise ValueError(f"domain must be in {VALID_DOMAINS} but was {domain}")
            if domain != "all" and version == "sentry" and domain not in SENTRY_DOMAINS:
                raise ValueError(f"domain must be in {SENTRY_DOMAINS} but was {domain}")
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be in {VALID_SPLITS} but was {split}")
        self._root_data_dir = root
        self._domain_list = domain_list
        self._split = split
        self._transform = transform
        self._version = version

        self._unlabeled = unlabeled
        self.data = load_dataset(root, domain_list, split, version)
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.228, 0.224, 0.225]
        if verbose:
            print(f'Loaded domains {", ".join(domain_list)}, split is {split}')
            print(f"Total number of images: {len(self.data)}")
            print(f"Total number of classes: {self.get_num_classes()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, y = self.data[idx]
        x = Image.open(os.path.join(self._root_data_dir, path))
        x = x.convert("RGB")
        if self._transform is not None:
            x = self._transform(x)
        # if self._unlabeled:
        #     return x, -1
        # else:
        return x, int(y)

    def get_num_classes(self):
        return len(set([self.data[idx][1] for idx in range(len(self.data))]))
