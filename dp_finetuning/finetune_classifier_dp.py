import os

import timm
import torch
import opacus
import numpy as np
import wandb
import torch.nn as nn
import torch.nn.functional as F

from opacus import PrivacyEngine
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from ema_pytorch import EMA
from tqdm import tqdm

from utils import parse_args, dataset_with_indices, extract_features

global args
global len_test
### UTILS
def train(args, model, device, train_loader, optimizer, privacy_engine, epoch, ema=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target, indices) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        violations = privacy_engine.accountant.get_violations(indices)
        real_step = optimizer.step(violations)
        privacy_engine.accountant.compute_norms(indices)
        if ema:
            ema.update() # update our EMA
        privacy_engine.accountant.step()
        print(f"Privacy Usage {args.epsilon * privacy_engine.accountant.privacy_usage.max().detach().item():.2f}")
        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

def do_test(model, data, target, criterion, test_loss, correct):
    output = model(data)
    test_loss += criterion(output, target).item()  # sum up batch loss
    pred = output.argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss/len_test, correct*100/len_test 

def test(model, device, test_loader, ema_model, swa_model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, test_loss_ema, correct_ema, test_loss_swa, correct_swa = 0,0,0,0,0,0
    with torch.no_grad():
        for data, target, _ in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            test_loss, correct = do_test(model, data, target, criterion, test_loss, correct)
            test_loss_ema, correct_ema = do_test(ema_model, data, target, criterion, test_loss_ema, correct_ema)
            test_loss_swa, correct_swa = do_test(swa_model, data, target, criterion, test_loss_swa, correct_swa)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss,correct))
    print("\nTest set (EMA): Average loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss_ema,correct_ema))
    print("\nTest set (SWA): Average loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss_swa,correct_swa))

    return max(correct, correct_ema)

# args = EasyDict({"dataset": "CIFAR10", "arch": "beitv2_large_patch16_224_in22k", 
#                  "num_classes":10, "lr": 8.0, "epochs": 16, "batch_size": 50000,
#                  "sigma": 50.0, "max_per_sample_grad_norm": 30, "delta": 1e-5,
#                  "secure_rng": False, "num_runs": 1, "seed": 11297, 
#                  "disable_dp": False, "device": "cuda:0", "epsilon": 0.1})
def main():
    ### ARGS
    global args
    global len_test

    args = parse_args()

    ### SET SEEDS
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ### EITHER FETCH EXTRACTED FEATURES OR EXTRACT FEATURES AND STORE THEM, THEN MAKE DATASET
    dataset_path = args.dataset_path + args.dataset
    abbrev_arch = args.arch.split("_")[0]
    extracted_path = args.dataset_path + "transfer/features/" + args.dataset + "_" + abbrev_arch
    extracted_train_path = extracted_path + "_train.npy"
    extracted_test_path = extracted_path + "_test.npy"

    if not os.path.exists(dataset_path):
        raise Exception('We cannot download a dataset/model here \n Run python utils.py to download things')
    ds = getattr(datasets, args.dataset.upper())(dataset_path, transform=transforms.ToTensor(), train=True)
    images_train, labels_train = torch.tensor(ds.data.transpose(0, 3, 1, 2)) / 255.0, torch.tensor(ds.targets)
    ds = getattr(datasets, args.dataset.upper())(dataset_path, transform=transforms.ToTensor(), train=False)
    images_test, labels_test = torch.tensor(ds.data.transpose(0, 3, 1, 2)) / 255.0, torch.tensor(ds.targets)
    len_test = labels_test.shape[0]

    if not os.path.exists(extracted_path):
        extract_features(args, images_train, images_test)

    x_train = np.load(extracted_train_path)
    features_train = torch.from_numpy(x_train)
    ds_train = dataset_with_indices(TensorDataset)(features_train, labels_train)
    x_test = np.load(extracted_test_path)
    ds_test = dataset_with_indices(TensorDataset)(torch.from_numpy(x_test), labels_test)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    # from opacus.utils.batch_memory_manager import wrap_data_loader
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    ### CREATE MODEL, OPTIMIZER AND MAKE PRIVATE
    classifier = nn.Linear(features_train.shape[-1], args.num_classes, bias=False).cuda()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0)
    privacy_engine = None

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        model, optimizer, train_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            clipping="budget",
            epsilon=args.epsilon,
            delta=args.delta,
            poisson_sampling=True,
            augmult=args.augmult,
        )
        # train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=10000, optimizer=optimizer)

    ### MAKE SOME AVERAGING UTILITES

    ema = EMA(model, 
                beta = 0.9999,
                update_after_step = 0,
                update_every = 1)
    from torch.optim.swa_utils import AveragedModel
    swa_model = AveragedModel(model)

    ### WANDB - COMMENT OUT IF YOU DON'T WANT TO USE WANDB

    wandb.init(project="dp_finetune", 
        entity="kiddyboots216")
    wandb.config.update(args)

    ### DO TRAINING
    corrects = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch, ema)
        new_correct = test(model, args.device, test_loader, ema, swa_model)
        corrects.append(new_correct)
        wandb.log({"test_acc": new_correct})
        if new_correct > 90:
            swa_model.update_parameters(model)
    best_acc = max(corrects)
    print(f"Best overall accuracy {best_acc:.2f}")
    wandb.log({"best_acc": best_acc,
                "epsilon": args.epsilon * privacy_engine.accountant.privacy_usage.max()})

if __name__ == "__main__":
    main()