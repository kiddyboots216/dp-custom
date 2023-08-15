import os

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

# from opacus.utils.batch_memory_manager import wrap_data_loader
from utils import (
    parse_args,
    get_ds,
)
from utils import set_all_seeds, ARCH_TO_NUM_FEATURES, DATASET_TO_SIZE
import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)


args = None
len_test = None
g_weight_cache = None
n_accum_steps = None

### UTILS
def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch_step, (data, target, _) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        loss = criterion(model(data), target)
        loss.backward()
        if ((epoch_step + 1) % n_accum_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    return np.mean(losses)


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
            # return val._module
            return val

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
    use_bias = False
    from timm.models.layers import trunc_normal_
    if args.dataset == "ImageNet":
        model = nn.Linear(ARCH_TO_NUM_FEATURES[args.arch][args.feature_mod], args.num_classes, bias=use_bias).cuda()
    else:
        model = nn.Linear(ARCH_TO_NUM_FEATURES[args.arch], args.num_classes, bias=use_bias).cuda()
    if args.standardize_weights: # by default this should be true, setting the weights to zero is very important
        # model.weight.data.normal_(mean=0.0, std=0.01)
        # model.bias.data.zero_()
        model.weight.data.zero_()
        # trunc_normal_(model.weight, std=0.01)
        if use_bias:
            # model[1].bias.data.add_(-10.)
            model.bias.data.add(-10.)
    
    from timm.optim import Lamb, AdamW
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=False,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=0.0)
    elif args.optimizer == "adam":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "pgd":

        class ProjectedGradientDescent(Optimizer):
            def __init__(self, params, lr=0.1, weight_decay=0, constraint=20):
                defaults = dict(lr=lr, weight_decay=weight_decay, constraint=constraint)
                super(ProjectedGradientDescent, self).__init__(params, defaults)

            def step(self, closure=None):
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        
                        # Perform regular gradient update
                        d_p = p.grad
                        p.data.add_(d_p, alpha=-group['lr'])

                        # Project updated weights onto L2-ball of specified radius
                        p.data.div_(max(1, p.data.norm() / group['constraint']))
        optimizer = ProjectedGradientDescent(model.parameters(), lr=args.lr, constraint=20)

    privacy_engine = None

    if not args.disable_dp:
        if args.privacy_engine == "opacus":
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng, accountant="rdp") # we don't actually use any accounting from opacus
            clipping_dict = {
                "vanilla": "flat",
                "individual": "budget",
                "dpsgdfilter": "filter",
                "sampling": "sampling",
            }
            clipping = clipping_dict[args.mode]
            model, optimizer, _ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                clipping=clipping,
                poisson_sampling=True,
            )
            # IF YOU RUN OUT OF MEMORY PLEASE DO NOT USE OPACUS IT REQUIRES SOME CHANGES TO THE CODE
            # if args.augmult > -1 or args.num_classes>10:
            #     print("WRAPPING DATA LOADER")
            #     train_loader = wrap_data_loader(
            #         data_loader=train_loader, max_batch_size=MAX_PHYS_BSZ, optimizer=optimizer
            #     )
        elif args.privacy_engine == "fastDP":
            from fastDP import PrivacyEngine
            privacy_engine = PrivacyEngine(
                module=model,
                batch_size=args.max_phys_bsz,
                sample_size=DATASET_TO_SIZE[args.dataset],
                epochs=args.epochs,
                max_grad_norm=args.max_per_sample_grad_norm,
                noise_multiplier=args.sigma,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                accounting_mode="glw",
                clipping_fn="Abadi",
                clipping_mode="MixOpt",
                clipping_style="all-layer",
                loss_reduction="mean",
            )
            privacy_engine.attach(optimizer)

    sched = None
    # if args.sched:
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    import torch.optim as optim
    class WarmupSchedule(optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_step, warmup_factor, lr, last_epoch=-1):
            self.warmup_step = warmup_step
            self.warmup_factor = warmup_factor
            self.lr = lr
            super(WarmupSchedule, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_step:
                return [self.lr for _ in self.base_lrs]
            return [base_lr * self.warmup_factor for base_lr in self.base_lrs]
    class CooldownSchedule(optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, decay_step, decay_factor, lr, last_epoch=-1):
            self.decay_step = decay_step
            self.decay_factor = decay_factor
            self.lr = lr
            super(CooldownSchedule, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.decay_step:
                return [self.lr for _ in self.base_lrs]
            return [base_lr / self.decay_factor for base_lr in self.base_lrs]

    if args.sched == '2':
        sched = CooldownSchedule(optimizer, decay_step=40, decay_factor=2, lr=args.lr)
    if args.sched == '1':
        sched = WarmupSchedule(optimizer, warmup_step=40, warmup_factor=2, lr=args.lr)
    # swa_model = AveragedModel(model)
    ema_avg = (
        lambda averaged_model_parameter, model_parameter, num_averaged: 0.1
        * averaged_model_parameter
        + 0.9 * model_parameter
    )
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    # ema_model = None
    return model, ema_model, optimizer, privacy_engine, sched, train_loader


def extract_grads_weights(model, raw_grads, noisy_grads, weights):
    """
    This function extracts the noisy grad, raw grad and weight from the model
    """
    raw_model = get_classifier({"model": model})[
        "model"
    ]  # get_classifier gets back the original model, unwraps it
    raw_grad = raw_model.weight.grad
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
        train_loss = train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)
        # only do test every 10 epochs, or at the end of training
        if epoch % 10 == 0 or epoch == args.epochs:
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
        ema_model.update_parameters(model)
    # extract weights from classifier
    # weights = model[1].weight.data
    weights = model.weight.data
    # print norm of weights
    print("NORM OF WEIGHTS", torch.norm(weights))        
    return (
        # torch.stack(raw_grads),
        # torch.stack(noisy_grads),
        # torch.stack(weights),
        None,
        None,
        weights,
        corrects,
    )


def store_grads(all_weights):
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
        final_weights = all_weights[-1]
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
    global n_accum_steps
    args = parse_args()
    args.max_phys_bsz = min(args.max_phys_bsz, args.batch_size)
    n_accum_steps = max(1, np.ceil(args.batch_size / args.max_phys_bsz))
    print("GETTING DATASET")
    train_loader, test_loader = get_ds(args)
    args.batch_size = args.max_phys_bsz
    len_test = len(test_loader.dataset)
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

        best_accs.append(max(corrects)) # using the best run is possible via fully adaptive composition
    store_grads(all_weights)

    # The below code is commented out because it is not discussed in the main body of the paper; but it is present in the appendix.
    # print("DOING ZERO-PRIVACY-COST TRAINING WITH MOMENTUM BUFFER")
    # for _ in range(10): # these iterations have no privacy cost!
    #     momentum_buffer = optimizer.state[optimizer.param_groups[0]['params'][0]]['momentum_buffer']
    #     model.weight.data.add_(momentum_buffer, alpha=(-1. * args.lr)) # hardcode fake iterations
    #     optimizer.state[optimizer.param_groups[0]['params'][0]]['momentum_buffer'].mul_(args.momentum).add_(momentum_buffer)
    #     new_correct = test(args, {
    #         "model": model,
    #     },
    #     args.device, test_loader)
    #     corrects.append(new_correct)
    best_acc = np.mean(best_accs)
    print(f"Mean accuracy {best_acc:.2f}")
    if args.num_runs > 1:
        print("ALL ACCS", best_accs)
        best_acc_std = np.std(best_accs)
        print(f"Std accuracy {best_acc_std:.2f}")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
