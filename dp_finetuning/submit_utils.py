from finetune_classifier_dp import *
from utils import *
from fastDP import PrivacyEngine
args = None
len_test = None
n_accum_steps = 1

def best_correct(test_stats):
    return max([test_stats[i] for i in test_stats if "_acc" in i])

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

def print_test_stats(test_stats):
    for key, val in test_stats.items():
        print(f"Test Set: {key} : {val:.4f}")

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

def do_training(
    model, ema_model, train_loader, test_loader, optimizer, privacy_engine, sched
):
    corrects = []
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
    return corrects

def setup_all(train_loader):
    ### CREATE MODEL, OPTIMIZER AND MAKE PRIVATE
    use_bias = False
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

    if args.privacy_engine == "opacus": # PLEASE DON'T USE OPACUS IT MIGHT ERROR
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
    ema_avg = (
        lambda averaged_model_parameter, model_parameter, num_averaged: 0.1
        * averaged_model_parameter
        + 0.9 * model_parameter
    )
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    # ema_model = None
    return model, ema_model, optimizer, privacy_engine, sched, train_loader

def launch_run(run_args, train_loader, test_loader):
    # if run_args.epsilon == 0.01:
    #     return np.random.choice([75, 85, 90])
    # elif run_args.epsilon == 0.05:
    #     return np.random.choice([94, 95, 96])
    # else:
    #     return 99
    global args
    global len_test
    args = run_args
    len_test = len(test_loader.dataset)
    model, ema_model, optimizer, privacy_engine, sched, train_loader = setup_all(
            train_loader
        )
    corrects = do_training(
            model,
            ema_model,
            train_loader,
            test_loader,
            optimizer,
            privacy_engine,
            sched,
        )
    return max(corrects)