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

from utils import (
    parse_args,
    dataset_with_indices,
    extract_features,
    get_ds,
    DATASET_TO_CLASSES,
    PiecewiseLinear,
)
from utils import set_all_seeds, ARCH_TO_INTERP_SIZE, ARCH_TO_NUM_FEATURES
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
    # losses = []
    for _, (data, target, _) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
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
                    optimizer.step(
                        indices, privacy_engine.accountant.get_violations(indices)
                    )
            else:
                if args.mode in ["dpsgdfilter"]:
                    optimizer.max_grad_norm = privacy_engine.accountant.max_grad_norm
                    real_step = optimizer.step()
                    if real_step:
                        privacy_engine.accountant.step()
                        print(
                            f"NORMS AT EPOCH {epoch}: {optimizer.max_grad_norm} -> {privacy_engine.accountant.max_grad_norm}"
                        )
                elif args.mode in ["individual", "sampling"]:
                    optimizer.compute_norms()
                    privacy_engine.accountant.compute_norms(indices)
                    optimizer.max_grad_norm = (
                        privacy_engine.accountant.max_grad_norm
                    )  # for sampling, does nothing
                    real_step = optimizer.step(
                        indices, privacy_engine.accountant.get_violations(indices)
                    )
                    if real_step:
                        privacy_engine.accountant.step()
                        print(
                            "PRIVACY USAGE SO FAR ",
                            privacy_engine.accountant.privacy_usage,
                        )
        # losses.append(loss.detach().numpy())

    # print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    # return np.mean(losses)
    return None


def do_test(model_dict, data, target, criterion, test_stats):
    for key, model in model_dict.items():
        model.eval()
        output = model(data)
        test_stats[key + "_loss"] += criterion(
            output, target
        ).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)
        test_stats[key + "_acc"] += (
            pred.eq(target.view_as(pred)).sum().item() * 100 / len_test
        )
    return test_stats


def get_classifier(model_dict):
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
        model_dict[key] = handle_augmult(
            args, key, handle_dp(args, key, handle_average(args, key, val))
        )
    return model_dict


def store_weights(args, model_dict, epoch):
    """
    Store weights of model
    """
    # global g_weight_cache
    model_dict = get_classifier(model_dict)
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
    model_dict = get_classifier(model_dict)
    test_stats = {key + "_loss": 0 for key in model_dict.keys()}
    test_stats.update({key + "_acc": 0 for key in model_dict.keys()})
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            test_stats = do_test(model_dict, data, target, criterion, test_stats)
    print_test_stats(test_stats)
    return best_correct(test_stats)


def setup_all(train_loader):
    ### CREATE MODEL, OPTIMIZER AND MAKE PRIVATE
    model = nn.Linear(
        ARCH_TO_NUM_FEATURES[args.arch], args.num_classes, bias=False
    ).cuda()
    # if args.augmult > -1:
    #     # model = timm.create_model(args.arch, num_classes=DATASET_TO_CLASSES[args.dataset], pretrained=True).cuda()
    #     feature_extractor = timm.create_model(args.arch, num_classes=0, pretrained=True).cuda()
    #     for p in feature_extractor.parameters():
    #     # for p in model.parameters():
    #         p.requires_grad = False
    #     # model.get_classifier().weight.requires_grad = True
    #     # model.get_classifier().bias.requires_grad = True
    #     model = nn.Sequential(feature_extractor,
    #                             model)
    #     model = nn.DataParallel(model)
    if args.standardize_weights:
        model.weight.data.zero_()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=False,
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng, accountant="gdp")
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
            # delta=args.delta,
            poisson_sampling=True,
        )
        # if args.augmult > -1 or args.num_classes>10:
        print("WRAPPING DATA LOADER")
        train_loader = wrap_data_loader(
            data_loader=train_loader, max_batch_size=10000, optimizer=optimizer
        )

    sched = None
    if args.sched:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # swa_model = AveragedModel(model)
    ema_avg = (
        lambda averaged_model_parameter, model_parameter, num_averaged: 0.1
        * averaged_model_parameter
        + 0.9 * model_parameter
    )
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    return model, ema_model, optimizer, privacy_engine, sched, train_loader


def extract_grads_weights(model, raw_grads, noisy_grads, weights):
    """
    This function extracts the noisy grad, raw grad and weight from the model
    """
    raw_model = get_classifier({"model": model})[
        "model"
    ]  # get_classifier gets back the original model, unwraps it
    raw_grad = raw_model.weight.summed_grad
    noisy_grad = raw_model.weight.grad  # p.grad = p.summed_grad + noise
    weight = raw_model.weight.data
    raw_grads.append(raw_grad)
    noisy_grads.append(noisy_grad)
    weights.append(weight)
    return raw_grads, noisy_grads, weights


def do_training(
    model, ema_model, train_loader, test_loader, optimizer, privacy_engine, sched
):
    raw_grads, noisy_grads, weights, corrects = [], [], [], []
    for epoch in range(1, args.epochs + 1):
        if sched is not None:
            sched.step()
            wandb.log({"lr": sched.get_lr()})
        train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)
        raw_grads, noisy_grads, weights = extract_grads_weights(
            model, raw_grads, noisy_grads, weights
        )
        new_correct = test(
            args,
            {
                "model": model,
                "ema": ema_model,
            },
            args.device,
            test_loader,
        )
        corrects.append(new_correct)
        wandb.log({"test_acc": new_correct})
        ema_model.update_parameters(model)
    return (
        torch.stack(raw_grads),
        torch.stack(noisy_grads),
        torch.stack(weights),
        corrects,
    )


def store_grads(all_noisy_grads, all_raw_grads, all_weights):
    if args.store_grads:
        all_noisy_grads, all_raw_grads, all_weights = (
            torch.stack(all_noisy_grads),
            torch.stack(all_raw_grads),
            torch.stack(all_weights),
        )
        f_dir = f"grad_datasets/{args.arch}/{args.dataset}"
        f_path = f"/grads_weights_{args.num_runs}_{args.epochs}_{int(args.lr)}_{int(args.epsilon)}"
        f_ext = ".npz"
        os.makedirs(f_dir, exist_ok=True)
        f_loc = f_dir + f_path + f_ext
        np.savez(
            f_loc,
            noisy_grads=all_noisy_grads.cpu().numpy(),
            raw_grads=all_raw_grads.cpu().numpy(),
            weights=all_weights.cpu().numpy(),
        )
        print(f"Saved grads to {f_loc}")
    elif args.store_weights:
        final_weights = all_weights[-1][-1]
        f_dir = f"ckpts/{args.arch}/{args.dataset}"
        f_path = (
            f"/weights_{args.num_runs}_{args.epochs}_{int(args.lr)}_{int(args.epsilon)}"
        )
        f_ext = ".npz"
        os.makedirs(f_dir, exist_ok=True)
        f_loc = f_dir + f_path + f_ext
        np.savez(f_loc, weights=final_weights.cpu().numpy())
        print(f"Saved weights to {f_loc}")


def main():
    global args
    global len_test
    args = parse_args()
    print("GETTING DATASET")
    train_loader, test_loader = get_ds(args)
    len_test = len(test_loader.dataset)
    wandb.init(project="baselines", entity="dp-finetuning")
    wandb.config.update(args)
    all_noisy_grads, all_raw_grads, all_weights, all_corrects = [], [], [], []
    best_accs = []
    for num_run in range(args.num_runs):
        print("SETTING SEEDS")
        set_all_seeds(args.seed + num_run)
        print("INITIALIZING EVERYTHING")
        model, ema_model, optimizer, privacy_engine, sched, train_loader = setup_all(
            train_loader
        )
        print("STARTING TRAINING")
        raw_grads, noisy_grads, weights, corrects = do_training(
            model,
            ema_model,
            train_loader,
            test_loader,
            optimizer,
            privacy_engine,
            sched,
        )
        all_noisy_grads.append(noisy_grads)
        all_raw_grads.append(raw_grads)
        all_weights.append(weights)
        all_corrects.append(corrects)
        best_accs.append(max(corrects))
    store_grads(all_noisy_grads, all_raw_grads, all_weights)

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
    wandb_dict = {"best_acc": best_acc, "best_acc_std": best_acc_std}
    logged_epsilon = None
    if args.mode in ["vanilla"] and args.epsilon != 0:
        logged_epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    elif args.mode in ["individual", "dpsgdfilter"]:
        logged_epsilon = args.epsilon * privacy_engine.accountant.privacy_usage.max()
    elif args.mode in ["sampling"]:
        logged_epsilon = privacy_engine.accountant.privacy_usage
    wandb_dict.update({"epsilon": logged_epsilon})
    wandb.log(wandb_dict)


if __name__ == "__main__":
    main()
