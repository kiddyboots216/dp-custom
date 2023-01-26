import os
import timm
import torch
import opacus
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from opacus.utils.batch_memory_manager import wrap_data_loader
import wandb

# from wilds import get_dataset
import torchvision.transforms as transforms

from utils import parse_args, ARCH_TO_INTERP_SIZE, ARCH_TO_NUM_FEATURES, set_all_seeds
from wilds_dataset import WILDS, WILDSTensor, Fmow, DomainNet, FmowTensor

args = None
criterion = nn.CrossEntropyLoss()
device = "cuda:0"

import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)

def load_wilds_ds():
    print("Loading dataset: {}".format(args.dataset))
    # Load the full dataset, and download it if necessary
    # dataset = get_dataset(dataset=args.dataset, download=True, root_dir=args.dataset_path)
    transform = transforms.ToTensor()
    if args.dataset == "fmow":
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        train_data = Fmow(regions=[3], split="train", root=args.dataset_path, transform=transform)
        val_data = Fmow(regions=[3], split="val", root=args.dataset_path, transform=transform)
        test_data = Fmow(regions=[1,2], split="val", root=args.dataset_path, transform=transform)
    elif args.dataset == "domainnet":
        train_data = DomainNet(domain='sketch', split='train', transform=transform,
                               unlabeled=False, verbose=True, version='full')
        test_data = DomainNet(domain='sketch', split='test', transform=transform,
                                unlabeled=False, verbose=True, version='full')
    else:
        train_data = WILDS(args.dataset, split="train", root=args.dataset_path, transform=transform)
        test_data = WILDS(args.dataset, split="test", root=args.dataset_path, transform=transform)
    bsz = 4
    if args.dataset in ["waterbirds", "domainnet"]:
        bsz = 1
    train_loader = DataLoader(train_data, batch_size=bsz, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=args.workers)
    # aux_test_datasets = [WILDS(args.dataset, split="test", root=args.dataset_path, transform=transform, meta_selector=meta_selector) for meta_selector in meta_selectors]
    # aux_test_loaders = [DataLoader(aux_test_dataset, batch_size=2, shuffle=False, num_workers=args.workers) for aux_test_dataset in aux_test_datasets]
    datasets = [train_data, test_data]
    # datasets += aux_test_datasets
    dataloaders = [train_loader, test_loader] 
    if args.dataset == "fmow":
        val_loader = DataLoader(val_data, batch_size=bsz, shuffle=False, num_workers=args.workers)
        datasets.append(val_data)
        dataloaders.append(val_loader)
    # dataloaders += aux_test_loaders
    return datasets, dataloaders
    # return train_data, test_data, train_loader, test_loader, dataset, aux_test_loaders

def gen_wilds_ds(loader, arch):
    feature_extractor = nn.DataParallel(timm.create_model(arch, num_classes=0, pretrained=True)).eval().to(device)
    
    def get_features(f, imgs, interp_size):
        features = []
        for img, label in tqdm(imgs):
            # img = batch[0]
            # MyPdb().set_trace()
            img = F.interpolate(img.to(device), size=(interp_size, interp_size), mode="bicubic")
            features.append(feature_extractor(img).detach().cpu())
        return torch.cat(features)
    
    interp_size = ARCH_TO_INTERP_SIZE[arch]
    # features_train = get_features(feature_extractor, train_loader, interp_size=interp_size)
    # features_test = get_features(feature_extractor, test_loader, interp_size=interp_size)
    features = get_features(feature_extractor, loader, interp_size=interp_size)
    return features

def get_features_paths():
    ### GET PATH
    extracted_paths = []
    extracted_path = args.dataset_path + "transfer/features/" + args.dataset.lower() + "_" + args.arch
    extracted_train_path = extracted_path + "/_train.npy"
    extracted_test_path = extracted_path + "/_test.npy"
    extracted_paths.append(extracted_train_path)
    extracted_paths.append(extracted_test_path)
    if args.dataset in ["fmow"]:
        extracted_val_path = extracted_path + "/_val.npy"
        extracted_paths.append(extracted_val_path)
    return extracted_paths

def get_wilds_ds():
    datasets, dataloaders = load_wilds_ds()
    # labels_train, labels_test = train_data._dataset.y_array, test_data._dataset.y_array
    paths = get_features_paths()
    if not all([os.path.exists(path) for path in paths]):
        os.makedirs(paths[0].rsplit("/", 1)[0], exist_ok=True)
        for path, dataloader in zip(paths, dataloaders):
            if not(os.path.exists(path)):
                features = gen_wilds_ds(loader=dataloader, arch=args.arch)
                np.save(path, features)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    feature_datasets = []
    if args.dataset == "fmow":
        # labels_train = datasets[0]._subset.y_array
        # labels_test = datasets[1]._subset.y_array
        labels_train, labels_test = None, None
    elif args.dataset == "domainnet":
        labels_train = np.array([int(d[1]) for d in datasets[0].data])
        labels_test = np.array([int(d[1]) for d in datasets[1].data])
    else:
        labels_train = datasets[0]._dataset.y_array
        labels_test = datasets[1]._dataset.y_array
    train_features = torch.from_numpy(np.load(paths[0]))
    # train_dataset = TensorDataset(train_features, labels_train)
    if args.dataset in ["fmow"]:
        train_dataset = FmowTensor(train_features, datasets[0])
    else:
        train_dataset = WILDSTensor(train_features, datasets[0], labels_train)
    feature_datasets += [train_dataset]
    
    test_features = torch.from_numpy(np.load(paths[1]))
    if args.dataset in ["fmow"]:
        test_dataset = FmowTensor(test_features, datasets[1])
        val_features = torch.from_numpy(np.load(paths[2]))
        val_dataset = FmowTensor(val_features, datasets[2])
        feature_datasets += [test_dataset]
        feature_datasets += [val_dataset]
    else:
        test_dataset = WILDSTensor(test_features, datasets[1], labels_test)
        feature_datasets += [test_dataset]
    len_f = lambda dataset: args.batch_size
    if args.batch_size == -1:
        len_f = lambda dataset: len(dataset)
    loaders = [DataLoader(dataset, batch_size=len_f(dataset), shuffle=True, **kwargs) for dataset in feature_datasets]
    return loaders

def get_model():
    model = nn.Linear(ARCH_TO_NUM_FEATURES[args.arch], args.num_classes, bias=False).to(device)
    model.weight.data.zero_()
    return model 

def get_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=args.weight_decay)
    return optimizer

def get_privacy_engine(model, loaders, optimizer):
    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = opacus.PrivacyEngine(secure_mode=False, accountant="gdp")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loaders[0],
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            clipping="flat",
            poisson_sampling=False,
        )
        if args.dataset == "domainnet":
            train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=2048, optimizer=optimizer)
    return model, optimizer, [train_loader] + loaders[1:], privacy_engine

def get_scheduler(optimizer):
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    return sched

def setup_all():
    print("Setting up all")
    loaders = get_wilds_ds()
    model = get_model()
    optimizer = get_optimizer(model)
    model, optimizer, loaders, privacy_engine = get_privacy_engine(model, loaders, optimizer)
    sched = get_scheduler(optimizer)
    return loaders, model, optimizer, privacy_engine, sched

def train(model, device, train_loader, optimizer):
    model.train()
    all_y_true = []
    all_y_pred = []
    # MyPdb().set_trace()
    for data, target in train_loader:
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        y_pred = output.argmax(dim=1, keepdim=True)
        all_y_pred.append(y_pred.cpu())
        all_y_true.append(target.cpu())
        loss = criterion(output, target)
        # print(loss)
        loss.backward()
        optimizer.step()
    # MyPdb().set_trace()
    return torch.cat(all_y_pred).squeeze(), torch.cat(all_y_true)
        
def test(model, device, test_loader):
    # _, loader = test_loader
    model.eval()
    all_y_true, all_y_pred = [], []

    for data, y_true in test_loader:
        data, y_true = data.to(device), y_true.to(device)
        y_pred = model(data).argmax(dim=1, keepdim=True)
        all_y_true.append(y_true.cpu())
        all_y_pred.append(y_pred.cpu())

    return torch.cat(all_y_pred).squeeze(), torch.cat(all_y_true)

def eval(loader, pred, true):
    correct = (pred == true).sum().item()/len(true)
    # name = loader.dataset.get_meta_selector()
    print(f"Accuracy: {correct:.4f}")
    return correct, correct
    
def wilds_eval(loader, pred, true):
    if args.dataset == "fmow":
        correct = (pred == true).sum().item()/len(true)
        region = loader.dataset._regions
        results_str = f"Accuracy on Region {region}: {correct:.4f}"
        results = {f"Accuracy_{region}": correct}
        # all_metadata = loader.dataset.dataset.subset_metadata
        # results, results_str = loader.dataset.dataset._subset.eval(pred, true, all_metadata)
    elif args.dataset == "domainnet":
        correct = (pred == true).sum().item()/len(true)
        results_str = f"Accuracy: {correct:.4f}"
        results = {"Accuracy": correct}
    else:
        all_metadata = loader.dataset.dataset._dataset.metadata_array[:pred.shape[0]]
        results, results_str = loader.dataset.dataset._dataset.eval(pred, true, all_metadata)
    print(results_str)
    return results

def update_test_results(best_test_results, test_results):
    """
    Best test results is 2 numbers, the average acc and worst group acc
    Update each of these based on test_results
    """
    # MyPdb().set_trace()
    if args.dataset == "domainnet":
        best_test_results[0] = max(best_test_results[0], test_results["Accuracy"])
        best_test_results[1] = max(best_test_results[1], test_results["Accuracy"])
    elif args.dataset == "waterbirds":
        best_test_results[0] = max(best_test_results[0], test_results['adj_acc_avg'])
        best_test_results[1] = max(best_test_results[1], test_results['acc_wg'])
    elif args.dataset == "fmow":
        best_test_results[0] = max(best_test_results[0], test_results["Accuracy_[3]"])
        best_test_results[1] = max(best_test_results[1], test_results["Accuracy_[1, 2]"])
        # best_test_results[0] = max(best_test_results[0], test_results['acc_avg'])
        # best_test_results[1] = max(best_test_results[1], test_results['acc_worst_region'])
    else:
        best_test_results[0] = max(best_test_results[0], test_results['acc_avg'])
        best_test_results[1] = max(best_test_results[1], test_results['acc_wg'])
    return best_test_results

def do_everything():
    print("Starting")
    loaders, model, optimizer, privacy_engine, sched = setup_all()
    best_test_results = [0,0]
    for i in range(args.epochs):
        # if args.dataset not in ["fmow"]:
        train_results = wilds_eval(loaders[0], *train(model, device, loaders[0], optimizer))
        # else:
        # train_results = eval(loaders[0], *train(model, device, loaders[0], optimizer))
        # if args.dataset not in ["fmow"]:
        test_results = wilds_eval(loaders[1], *test(model, device, loaders[1]))
        if args.dataset in ["fmow"]:
            val_results = wilds_eval(loaders[2], *test(model, device, loaders[2]))
            test_results.update(val_results)
        best_test_results = update_test_results(best_test_results, test_results)
    return best_test_results, model, privacy_engine

def log_wandb(best_accs, model, privacy_engine):
    best_accs = np.array(best_accs)
    adj_acc_avg = best_accs[:,0].mean()
    acc_wg = best_accs[:,1].mean()
    adj_acc_std = best_accs[:,0].std()
    acc_wg_std = best_accs[:,1].std()
    wandb_dict = {
        "adj_acc_avg": adj_acc_avg,
        "acc_wg": acc_wg,
        "adj_acc_std": adj_acc_std,
        "acc_wg_std": acc_wg_std,
    }
    logged_epsilon = None
    if not args.disable_dp and args.epsilon != 0:
        logged_epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    wandb_dict.update({"epsilon": logged_epsilon})
    wandb.log(wandb_dict)

def main():
    global args 
    args = parse_args()
    wandb.init(project="baselines", 
        entity="dp-finetuning")
    wandb.config.update(args)
    best_accs = []
    model, privacy_engine = None, None
    for num_run in range(args.num_runs):
        set_all_seeds(args.seed + num_run)
        best_acc, model, privacy_engine = do_everything()
        best_accs.append(best_acc)
    log_wandb(best_accs, model, privacy_engine)
        
if __name__ == "__main__":
    main()