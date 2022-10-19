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
from tqdm import tqdm

from utils import parse_args, dataset_with_indices, extract_features, get_ds

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
                optimizer.step()
            else:
                optimizer.compute_norms(privacy_engine.accountant.get_violations(indices))
                privacy_engine.accountant.compute_norms(indices)
                real_step = optimizer.step()
                if real_step:
                    privacy_engine.accountant.step()
        # print(f"Privacy Usage {args.epsilon * privacy_engine.accountant.privacy_usage.max().detach().item():.2f}")
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

    return max(correct, correct_ema, correct_swa)

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
    if args.standardize_weights:
        nn.init.normal_(model.weight, mean=0.0, std=1.0/np.sqrt(num_features))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    privacy_engine = None

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
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
        # train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=10000, optimizer=optimizer)

    ### MAKE SOME AVERAGING UTILITES

    from torch.optim.swa_utils import AveragedModel
    swa_model = AveragedModel(model)
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    # sched = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, args.lr, step_size_up=10)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    ### WANDB - COMMENT OUT IF YOU DON'T WANT TO USE WANDB

    wandb.init(project="baselines", 
        entity="dp-finetuning")
    wandb.config.update(args)

    ### DO TRAINING
    corrects = []
    for epoch in range(1, args.epochs + 1):
        sched.step()
        train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)
        new_correct = test(model, args.device, test_loader, ema_model, swa_model)
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