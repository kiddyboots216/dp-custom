import os
import timm
import torch
import opacus
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import easydict
from prv_accountant.dpsgd import find_noise_multiplier
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from opacus.utils.batch_memory_manager import wrap_data_loader

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms

from utils import parse_args

criterion = nn.CrossEntropyLoss()
device = "cuda:0"
ARCH_TO_INTERP_SIZE = {"beit_large_patch16_512": 512,
        "convnext_xlarge_384_in22ft1k": 384,
        "beitv2_large_patch16_224_in22k": 224}
def load_wilds_ds(dataset_name, 
                  root_dir="/data/nvme/ashwinee/datasets"):
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
    )

    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=4)
    
    # Get the training set
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
    )

    # Prepare the standard data loader
    test_loader = get_eval_loader("standard", test_data, batch_size=4)
    return train_data, test_data, train_loader, test_loader, dataset

def gen_wilds_ds(train_loader, test_loader, arch="beitv2_large_patch16_224_in22k"):
    feature_extractor = nn.DataParallel(timm.create_model(arch, num_classes=0, pretrained=True)).eval().cuda()
    
    def get_features(f, imgs, interp_size):
        features = []
        for batch in tqdm(imgs):
            img = batch[0]
            img = F.interpolate(img.cuda(), size=(interp_size, interp_size), mode="bicubic")
            features.append(feature_extractor(img).detach().cpu())
        return torch.cat(features)
    
    interp_size = ARCH_TO_INTERP_SIZE[arch]
    features_train = get_features(feature_extractor, train_loader, interp_size=interp_size)
    features_test = get_features(feature_extractor, test_loader, interp_size=interp_size)
    return features_train, features_test
    
def get_wilds_ds(args):
    ### GET DATA
    dataset_path = args.dataset_path + args.dataset
    # we need to keep around the original test_loader so that we can use the eval function
    train_data, test_data, train_loader, test_loader, dataset = load_wilds_ds(args.dataset, root_dir=args.dataset_path)
    labels_train, labels_test = train_data.y_array, test_data.y_array
    ### GET PATH
    abbrev_arch = args.arch
    extracted_path = args.dataset_path + "transfer/features/" + args.dataset.lower() + "_" + abbrev_arch
    extracted_train_path = extracted_path + "/_train.npy"
    extracted_test_path = extracted_path + "/_test.npy"
    if not os.path.exists(extracted_path):
        features_train, features_test = gen_wilds_ds(train_loader,
                                                     test_loader,
                                                 args.arch)
        os.makedirs(extracted_path, exist_ok=True)
        np.save(extracted_train_path, features_train)
        np.save(extracted_test_path, features_test)  
    kwargs = {'num_workers': 1, 'pin_memory': True}
    x_train = np.load(extracted_train_path)
    features_train = torch.from_numpy(x_train)
    ds_train = TensorDataset(features_train, labels_train)
    args.batch_size = len(ds_train)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    x_test = np.load(extracted_test_path)
    features_test = torch.from_numpy(x_test)
    ds_test = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, **kwargs)
    return train_loader, test_loader, features_test.shape[-1], len(labels_test), dataset, test_data

def setup_all(args):
    train_loader, test_loader, num_features, len_test, dataset, test_data = get_wilds_ds(args)
    model = nn.Linear(num_features, args.num_classes, bias=False).cuda()
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = opacus.PrivacyEngine(secure_mode=False, accountant="gdp")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=1,
            clipping="flat",
            poisson_sampling=True,
        )
    return model, optimizer, privacy_engine, train_loader, test_loader, num_features, len_test, dataset, test_data

def train(args, model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
def test(args, model, device, test_loader, dataset, test_data):
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_metadata = []

    for data, y_true in test_loader:
        data, y_true = data.to(device), y_true.to(device)
        y_pred = model(data).argmax(dim=1, keepdim=True)
        all_y_true.append(y_true.cpu())
        all_y_pred.append(y_pred.cpu())

    all_y_true = torch.cat(all_y_true)
    all_y_pred = torch.cat(all_y_pred).squeeze()

    all_metadata = test_data.metadata_array
    results, results_str = dataset.eval(all_y_pred, all_y_true, all_metadata)
    print(results_str)
    return results

def do_everything(args):
    model, optimizer, privacy_engine, train_loader, test_loader, num_features, len_test, dataset, test_data = setup_all(args)
    for i in range(args.epochs):
        train(args, model, device, train_loader, optimizer)
        test(args, model, device, test_loader, dataset, test_data)
    if not args.disable_dp:
        print(privacy_engine.accountant.get_epsilon(delta=1e-5))
        
def main():
    args = parse_args()
    do_everything(args)
        
if __name__ == "__main__":
    main()