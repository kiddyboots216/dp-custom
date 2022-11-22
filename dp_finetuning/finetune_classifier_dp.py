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

from utils import parse_args, dataset_with_indices, extract_features, get_ds, DATASET_TO_CLASSES, PiecewiseLinear

args = None
len_test = None
g_weight_cache = None

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
        if args.disable_dp or args.mode in ["vanilla"]:
            optimizer.step()
        else:
            # this is quite messy - consider refactoring
            if len(train_loader) == 1:
                if args.mode in ["dpsgdfilter"]:
                    privacy_engine.accountant.step()
                    optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm
                    optimizer.step()
                elif args.mode in ["individual", "sampling"]:
                    optimizer.compute_norms()
                    privacy_engine.accountant.compute_norms(indices)
                    privacy_engine.accountant.step()
                    optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm
                    optimizer.step(indices, privacy_engine.accountant.get_violations(indices))
            else:
                if args.mode in ["dpsgdfilter"]:
                    optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm
                    real_step = optimizer.step()
                    if real_step:
                        privacy_engine.accountant.step()
                        print(f"NORMS AT EPOCH {epoch}: {optimizer.max_grad_norm} -> {privacy_engine.accountant.max_grad_norm}")
                elif args.mode in ["individual", "sampling"]:
                    optimizer.compute_norms()
                    privacy_engine.accountant.compute_norms(indices)
                    optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm # for sampling, does nothing
                    real_step = optimizer.step(indices, privacy_engine.accountant.get_violations(indices))
                    if real_step:
                        privacy_engine.accountant.step()
                        print("PRIVACY USAGE SO FAR ", privacy_engine.accountant.privacy_usage)
        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    return np.mean(losses)

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
    def handle_augmult(args, key, val):
        if args.augmult > -1:
            return val[1]
        else:
            return val
    for key, val in model_dict.items():
        model_dict[key] = handle_augmult(args, key, 
                                         handle_dp(args, key, 
                                                   handle_average(args, key, val)))
    return model_dict

def store_weights(args, model_dict, epoch):
    """
    Store weights of model
    """
    global g_weight_cache
    model_dict = get_classifier(args, model_dict)
    for key, val in model_dict.items():
        model_dict[key] = val.weight.detach().cpu()
    if g_weight_cache is None:
        g_weight_cache = torch.zeros(args.epochs+1, model_dict["model"].shape.numel()) # assumes sample_rate = 1
    g_weight_cache[epoch-1, :] = model_dict["model"].flatten()
    
def weights_to_grads(weight_cache):
    """
    Takes buffer of size (n_models, model_size) and turns it into a buffer of size (n_grads = n_models-1, model_size)
    """
    grad_cache = weight_cache[1:, :] - weight_cache[:1, :]
    return grad_cache

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
        model = nn.DataParallel(model)
    if args.standardize_weights:
        nn.init.normal_(model.weight, mean=0.0, std=1.0/np.sqrt(num_features))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)
    privacy_engine = None

    # if not args.disable_dp:
    privacy_engine = PrivacyEngine(secure_mode=args.secure_rng,
                                    accountant="gdp")
    clipping_dict = {
        "vanilla": "flat",
        "individual": "budget",
        "dpsgdfilter": "filter",
        "sampling": "sampling",
    }
    clipping = clipping_dict[args.mode]
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        clipping=clipping,
        epsilon=args.epsilon,
        delta=args.delta,
        poisson_sampling=True,
        augmult=args.augmult,
        expected_sample_rate=args.sample_rate,
    )
    if args.augmult > -1 or args.num_classes>10:
        train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=10000, optimizer=optimizer)

    print("TRAIN LOADER LEN", len(train_loader))
    ### MAKE SOME AVERAGING UTILITES

    swa_model = AveragedModel(model)
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    sched = None
    if args.sched:
        # lr_schedule = PiecewiseLinear([0, 10, args.epochs],
        #                           [0.1, args.lr,                  0.1])
        # lambda_step = lambda step: lr_schedule(step)
        # sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_step)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr = args.lr, 
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=int(1 * args.epochs),
                                                    )
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
    import pdb
    import code

    class MyPdb(pdb.Pdb):
        def do_interact(self, arg):
            code.interact("*interactive*", local=self.curframe_locals)
    # MyPdb().set_trace()
    # reference = model._module.weight
    # noise = torch.normal(
    #     mean=0,
    #     std=args.sigma,
    #     size=reference.shape,
    #     device=reference.device,
    #     generator=generator,
    # )
    for epoch in range(1, args.epochs + 1):
        store_weights(args, {
            "model": model,
            "ema": ema_model,
            "swa": swa_model,
        }, epoch)
        if sched is not None:
            sched.step()
            wandb.log({"lr" : sched.get_lr()})
        train_loss = train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)   
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
        # if args.do_vanilla:
        #     epsilon_reported = privacy_engine.accountant.get_epsilon(delta=1e-5)
        #     if epsilon_reported >= args.epsilon:
        #         break
    store_weights(args, {
            "model": model,
            "ema": ema_model,
            "swa": swa_model,
        }, args.epochs + 1)
    grad_cache = weights_to_grads(g_weight_cache)
    print("DOING FAKE TRAINING WITH MOMENTUM BUFFER")
    for _ in range(10):
        momentum_buffer = optimizer.original_optimizer.state[optimizer.original_optimizer.param_groups[0]['params'][0]]['momentum_buffer']
        model._module.weight.data.add_(momentum_buffer, alpha=-1) # hardcode fake iterations
        optimizer.original_optimizer.state[optimizer.original_optimizer.param_groups[0]['params'][0]]['momentum_buffer'].mul_(args.momentum).add_(momentum_buffer)
        new_correct = test(args, {
            "model": model,
        }, 
        args.device, test_loader)
        corrects.append(new_correct)
    
    best_acc = max(corrects)
    print(f"Best overall accuracy {best_acc:.2f}")
    if args.disable_dp:
        wandb.log({"best_acc": best_acc})
    elif args.mode in ["vanilla"]:
        wandb.log({"best_acc": best_acc,
                "epsilon": privacy_engine.accountant.get_epsilon(delta=1e-5)})
    elif args.mode in ["individual", "dpsgdfilter"]:
        wandb.log({"best_acc": best_acc,
                "epsilon": args.epsilon * privacy_engine.accountant.privacy_usage.max()})
    elif args.mode in ["sampling"]:
        wandb.log({"best_acc": best_acc,
                   "epsilon": privacy_engine.accountant.privacy_usage})

if __name__ == "__main__":
    main()