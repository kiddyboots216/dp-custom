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
import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)
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
    # global g_weight_cache
    model_dict = get_classifier(args, model_dict)
    for key, val in model_dict.items():
        model_dict[key] = val.weight.detach().cpu()
    # if g_weight_cache is None:
        # g_weight_cache = torch.zeros(args.epochs+1, model_dict["model"].shape.numel()) # assumes sample_rate = 1
    # g_weight_cache[epoch-1, :] = model_dict["model"].flatten()
    return model_dict["model"].flatten()
    
# def weights_to_grads(weight_cache):
#     """
#     Takes buffer of size (n_models, model_size) and turns it into a buffer of size (n_grads = n_models-1, model_size)
#     """
#     print(f"LIST OF WEIGHT NORMS {torch.norm(weight_cache, dim=1)}")
#     # MyPdb().set_trace()
#     grad_cache = weight_cache[1:, :] - weight_cache[:1, :]
#     return grad_cache

def best_correct(test_stats):
    return max([test_stats[i] for i in test_stats if "_acc" in i]) 

def print_test_stats(test_stats):
    for key, val in test_stats.items():
        print(f"Test Set: {key} : {val:.4f}")
        # wandb.log({key: val})

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

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

def setup_all():
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
        model.weight.data.zero_()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=False)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None

    if not args.disable_dp:
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
            delta=args.delta,
            poisson_sampling=True,
        )
        if args.augmult > -1 or args.num_classes>10:
            train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=5000, optimizer=optimizer)

    swa_model = AveragedModel(model)
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    return model, ema_model, swa_model, optimizer, privacy_engine, sched

def binary_search(initial_guess, min_val, max_val, num_tries=10):
    # do a binary search over values of r starting from the initial guess to find the value that gives the best accuracy
    best_r_so_far = initial_guess
    next_r = initial_guess
    best_acc_so_far = 0
    for i in range(num_tries):
        hparams = get_hparams(next_r)
        result = train_and_test(hparams)
        if result["test_acc"] > best_acc_so_far:
            best_acc_so_far = result["test_acc"]
            best_r_so_far = next_r

def extract_grads_weights(model, raw_grads, noisy_grads, weights):
    raw_grad = layer.
    raw_grads.append(raw_grad)
    noisy_grads.append(noisy_grad)
    weights.append(weight)

def do_training(model, ema_model, optimizer, privacy_engine, sched)
    raw_grads, noisy_grads, weights = [], []
    for epoch in range(1, args.epochs + 1):
        if sched is not None:
            sched.step()
            wandb.log({"lr" : sched.get_lr()})
        train_loss = train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)   
        raw_grads, noisy_grads, weights = extract_grads_weights(model)
        new_correct = test(args, {"model": model,"ema": ema_model,}, args.device, test_loader)
        corrects.append(new_correct)
        wandb.log({"test_acc": new_correct})
        ema_model.update_parameters(model)
    return torch.stack(weights)

def main():
    global args
    global len_test
    args = parse_args()
    train_loader, test_loader, num_features, len_test = get_ds(args)
    wandb.init(project="baselines", 
        entity="dp-finetuning")
    wandb.config.update(args)
    grads, weights = [], []
    best_accs = []
    for num_run in range(args.num_runs):
        set_all_seeds(args.seed + num_run)
        model, ema_model, optimizer, privacy_engine, sched = setup_all()
        run_weights = do_training(model, ema_model, optimizer, privacy_engine, sched)
        grad_cache = weights_to_grads(run_weights)
        grads.append(torch.stack(grad_cache))
        weights.append(torch.stack(g_weight_cache))
        g_weight_cache = []
        best_accs.append(max(corrects))
    if args.store_grads:
        grads, weights = torch.stack(grads), torch.stack(weights)
        np.savez(f"grad_datasets/{args.arch}/{args.dataset}/grads_weights_{args.num_runs}_{args.epochs}_{args.lr}.npz", 
                grad=grads.numpy(), weight=weights.numpy())
    
    # print("VERIFYING THAT CHANGING THE WEIGHTS BY CONSTANT FACTOR DOES NOT CHANGE TEST ACC")
    # model._module.weight.data.zero_()
    # model._module.weight.data.add_(grad_cache[-1].reshape(model._module.weight.data.shape).cuda() * 0.01)
    # test(args, {
    #         "model": model,
    #     }, 
    # args.device, test_loader)
    # print("DOING FAKE TRAINING WITH MOMENTUM BUFFER")
    # momentum_buffer = optimizer.original_optimizer.state[optimizer.original_optimizer.param_groups[0]['params'][0]]['momentum_buffer']
    # model._module.weight.data.add_(momentum_buffer, alpha=(-1. * args.lr)) # hardcode fake iterations
    # optimizer.original_optimizer.state[optimizer.original_optimizer.param_groups[0]['params'][0]]['momentum_buffer'].mul_(args.momentum).add_(momentum_buffer)
    # new_correct = test(args, {
    #     "model": model,
    # }, 
    # args.device, test_loader)
    # corrects.append(new_correct)
    
    best_acc = np.mean(best_accs)
    print(f"Best overall accuracy {best_acc:.2f}")
    best_acc_std = np.std(best_accs)
    if args.disable_dp or args.sigma==0:
        wandb.log({"best_acc": best_acc, "best_acc_std": best_acc_std})
    elif args.mode in ["vanilla"]:
        wandb.log({"best_acc": best_acc,
                   "best_acc_std": best_acc_std,
                "epsilon": privacy_engine.accountant.get_epsilon(delta=1e-5)})
    elif args.mode in ["individual", "dpsgdfilter"]:
        wandb.log({"best_acc": best_acc,
                "epsilon": args.epsilon * privacy_engine.accountant.privacy_usage.max()})
    elif args.mode in ["sampling"]:
        wandb.log({"best_acc": best_acc,
                   "epsilon": privacy_engine.accountant.privacy_usage})

if __name__ == "__main__":
    main()