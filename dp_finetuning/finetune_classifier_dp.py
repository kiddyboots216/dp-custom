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

from utils import parse_args, dataset_with_indices, extract_features, get_ds, DATASET_TO_CLASSES

global args
global len_test
### UTILS
def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target, indices) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # get per sample norms
        # step accountant
        # update optimizer max grad norm with accountant max grad norm
        # step optimizer
        if args.disable_dp or args.do_vanilla:
            optimizer.step()
        else:
            if len(train_loader) == 1:
                optimizer.compute_norms(privacy_engine.accountant.get_violations(indices))
                privacy_engine.accountant.compute_norms(indices)
                privacy_engine.accountant.step()
                optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm
                optimizer.step(indices)
            else:
                optimizer.compute_norms(privacy_engine.accountant.get_violations(indices))
                privacy_engine.accountant.compute_norms(indices)
                optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm
                real_step = optimizer.step(indices)
                if real_step:
                    privacy_engine.accountant.step()
        # print(f"Privacy Usage {args.epsilon * privacy_engine.accountant.privacy_usage.max().detach().item():.2f}")
        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

def do_test(model_dict, data, target, criterion, test_stats):
    for key, model in model_dict.items():
        model.eval()
        output = model(data)
        test_stats[key + "_loss"] += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)
        test_stats[key + "_acc"] += pred.eq(target.view_as(pred)).sum().item() * 100/len_test
    return test_stats

def get_classifier(args, model_dict):
    """
    this is a catastrophe
    """
    if args.augmult > -1:
        for key, val in model_dict.items():
            def handle_dp(args, key, val):
                if args.disable_dp:
                    return val
                else:
                    return val._module
            def handle_average(args, key, val):
                if key in ["ema", "swa"]:
                    return val.module
                else:
                    return val
            model_dict[key] = handle_dp(args, 
                                            key, 
                                            handle_average(args, 
                                                    key, 
                                                    val))[1]
                                                    # val)).get_classifier()
    return model_dict

def best_correct(test_stats):
    return max([test_stats[i] for i in test_stats if "_acc" in i]) 

def print_test_stats(test_stats):
    for key, val in test_stats.items():
        print(f"Test Set: {key} : {val:.4f}")

def test(args, model_dict, device, test_loader):
    model_dict = get_classifier(args, model_dict)
    test_stats = {key + "_loss": 0 for key in model_dict.keys()}
    test_stats.update({key + "_acc": 0 for key in model_dict.keys()})
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target, _ in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            test_stats = do_test(model_dict, data, target, criterion, test_stats)
    print_test_stats(test_stats)
    return best_correct(test_stats)

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
    train_loader, test_loader, num_features, len_test = get_ds(args)

    ### CREATE MODEL, OPTIMIZER AND MAKE PRIVATE
    model = nn.Linear(num_features, args.num_classes, bias=False).cuda()
    if args.augmult > -1:
        # model = timm.create_model(args.arch, num_classes=DATASET_TO_CLASSES[args.dataset], pretrained=True).cuda()
        feature_extractor = timm.create_model(args.arch, num_classes=0, pretrained=True).cuda()
        for p in feature_extractor.parameters():
        # for p in model.parameters():
            p.requires_grad = False
        # model.get_classifier().weight.requires_grad = True
        # model.get_classifier().bias.requires_grad = True
        model = nn.Sequential(feature_extractor,
                                model)
        # model = nn.DataParallel(model)
    if args.standardize_weights:
        nn.init.normal_(model.weight, mean=0.0, std=1.0/np.sqrt(num_features))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    privacy_engine = None

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng,
                                        accountant="gdp")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            clipping="flat" if args.do_vanilla else "budget",
            epsilon=args.epsilon,
            delta=args.delta,
            poisson_sampling=True,
            augmult=args.augmult,
        )
        if args.augmult > -1:
            train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=250, optimizer=optimizer)

    print("TRAIN LOADER LEN", len(train_loader))
    ### MAKE SOME AVERAGING UTILITES

    swa_model = AveragedModel(model)
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    sched = None
    # sched = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, args.lr, step_size_up=10)
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    ### WANDB - COMMENT OUT IF YOU DON'T WANT TO USE WANDB

    wandb.init(project="baselines", 
        entity="dp-finetuning")
    wandb.config.update(args)

    ### DO TRAINING
    corrects = []
    # corrects.append(test(args, {
    #         "model": model,
    #         "ema": ema_model,
    #         "swa": swa_model,
    #     }, args.device, test_loader)) # ensure we can test
    for epoch in range(1, args.epochs + 1):
        if sched is not None:
            sched.step()
        train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)
        new_correct = test(args, {
            "model": model,
            "ema": ema_model,
            "swa": swa_model,
        }, 
        args.device, test_loader)
        corrects.append(new_correct)
        wandb.log({"test_acc": new_correct})
        # update ema / swa
        ema_model.update_parameters(model)
        if (len(corrects) > 10 and (abs(corrects[-1] - corrects[-2]) < 1)): # model is converging, let's start doing SWA
            swa_model.update_parameters(model)
    best_acc = max(corrects)
    print(f"Best overall accuracy {best_acc:.2f}")
    if args.disable_dp:
        wandb.log({"best_acc": best_acc})
    elif args.do_vanilla:
        wandb.log({"best_acc": best_acc,
                "epsilon": privacy_engine.accountant.get_epsilon(delta=1e-5)})
    else:
        wandb.log({"best_acc": best_acc,
                "epsilon": args.epsilon * privacy_engine.accountant.privacy_usage.max()})

if __name__ == "__main__":
    main()