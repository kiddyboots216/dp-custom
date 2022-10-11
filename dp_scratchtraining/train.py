import os
import shutil
import time
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import models

import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import wrap_data_loader, BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import torch.distributed as dist
import logging

from ema_pytorch import EMA
import torchvision
from augmult import DatasetLoader, my_collate_func
from torch.utils.data.dataloader import default_collate
import torch.multiprocessing as mp

import wandb

from utils import parse_args, PiecewiseLinear, AverageMeter, accuracy, dataset_with_indices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# util function so we don't have to subclass every new dataset


# used for logging to TensorBoard
#from tensorboard_logger import configure, log_value

best_prec1 = 0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
global max_physical_batch_size


def setup():
    # wandb.run.log_code("/home/ashwinee/opacus/opacus/")
    global args, best_prec1
    args = parse_args() # returns parser.parse_args()
    wandb.init(project="dp", 
        entity="kiddyboots216")
    wandb.config.update(args)
    global max_physical_batch_size
    max_physical_batch_size = 65
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.do_augment:
        if args.model == "WideResNet":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Pad(4,padding_mode='reflect'),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                    (4,4,4,4),mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif args.model == "ResNet9":
            # equals np.mean(train_set.train_data, axis=(0,1,2))/255	
            cifar10_mean = (0.4914, 0.4822, 0.4465)	
            # equals np.std(train_set.train_data, axis=(0,1,2))/255	
            cifar10_std = (0.2471, 0.2435, 0.2616)	

            transform_train = transforms.Compose([	
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),	
                    transforms.RandomHorizontalFlip(),	
                    transforms.ToTensor(),	
                    transforms.Normalize(cifar10_mean, cifar10_std)	
                ])	
    elif args.do_augmult:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    if args.do_augmult:
        train_dataset = DatasetLoader(
            root='datasets/cifar10', 
            train=True, 
            download=True,
            transform=transform_train,
            image_size=(3, 32, 32),
            augmult=args.augmult,
            random_flip=True,
            random_crop=True,
            crop_size=32,
            pad=4,
        )

        train_loader = DataLoader(train_dataset,
                                  collate_fn=my_collate_func, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  **kwargs)
    else:
        train_dataset = dataset_with_indices(datasets.__dict__[args.dataset.upper()])(
            'datasets/cifar10', 
            train=True, 
            download=True,
            transform=transform_train
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            shuffle=True, 
            **kwargs
        )
    accounting_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 1*int(0.04* len(train_dataset)),
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )
    val_loader = DataLoader(
        datasets.__dict__[args.dataset.upper()]('datasets/cifar10', train=False, transform=transform_test),
        batch_size=256, shuffle=True, **kwargs)

    # create model
    model_cls = getattr(models, args.model)
    if args.model == "WideResNet":
        model = model_cls(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate, use_ws=True)
    elif args.model == "ResNet9":
        model = model_cls(num_classes=10, use_ws=True)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)
    if args.model == "WideResNet":
        lr_to_use = args.lr
    elif args.model == "ResNet9":
        lr_to_use = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr_to_use,
                                momentum=args.momentum, nesterov = (args.nesterov if args.momentum > 0 else False),
                                weight_decay=args.weight_decay)
    privacy_engine = None
    privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
        )
    
    if args.model == "WideResNet":
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    elif args.model == "ResNet9":
        scheduler = None
        # lr_schedule = PiecewiseLinear([0, int(args.epochs * 5/24), args.epochs],
        #                           [0, args.lr,                  0])
        # spe = np.ceil(len(train_loader.dataset) / max_physical_batch_size)
        # lambda_step = lambda step: lr_schedule(step / spe)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_step)
    return model, optimizer, train_loader, val_loader, privacy_engine, accounting_loader, scheduler 



def main():
    global best_prec1
    criterion = nn.CrossEntropyLoss()
    
    model, optimizer, train_loader, val_loader, privacy_engine, accounting_loader, scheduler = setup()    
    # if args.clip_per_layer:
    #     # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
    #     n_layers = len(
    #         [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    #     )
    #     max_grad_norm = [
    #         args.max_per_sample_grad_norm / np.sqrt(n_layers)
    #     ] * n_layers
    # else:
    max_grad_norm = args.max_per_sample_grad_norm
    clipping = "per_layer" if args.clip_per_layer else "flat"
    clipping = "topk" if args.do_topk else clipping
    prepend = "budget" if args.do_budget else ""
    clipping += prepend
    if clipping == "flat":
        model = opacus.grad_sample.grad_sample_module.GradSampleModule(
                model, batch_first=True, loss_reduction="mean", strict=False,
            )
        model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=max_grad_norm,
        clipping=clipping,
        poisson_sampling=True)
    else:
        clipping = "flat" if clipping == "per_layer" else clipping
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            k=args.k,
            do_induced=args.do_induced,
            do_baseline=args.do_baseline,
            n_stale=args.n_stale,
            accounting_loader=accounting_loader,
            criterion=criterion,
            device=device,
            epsilon=args.epsilon,
            delta=1e-5,
            poisson_sampling=False,
            augmult=args.augmult if args.do_augmult else 0,
        )
    train_loader = wrap_data_loader(
        data_loader=train_loader, 
        max_batch_size=max_physical_batch_size,
        optimizer=optimizer)
    ema = EMA(model, 
              beta = 0.9999,
              update_after_step = 0,
              update_every = 1)
    ema = ema.to(device)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, ema, criterion, optimizer, epoch, privacy_engine, scheduler)

        # evaluate on validation set
        prec1 = validate(val_loader, model, ema, criterion, epoch, privacy_engine) # pass privacy engine so we can compute privacy stats
        wandb.log({"prec1": prec1})
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)
    if args.do_budget:
        privacy_spent = privacy_engine.accountant.privacy_usage.cpu()
        privacy_artifact = wandb.Table(dataframe=pd.DataFrame(privacy_spent))
    wandb.log({"privacy": privacy_artifact})
    wandb.log({"Best Prec1": best_prec1})

def train(train_loader, model, ema, criterion, optimizer, epoch, privacy_engine, scheduler, rank=0):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    ema.train()

    end = time.time()
    for i, (data, target, indices) in enumerate(train_loader):
        # target = target.cuda(non_blocking=True)
        # data = data.cuda(non_blocking=True)
        data, target, indices = data.to(device), target.to(device), indices.cpu()

        # compute output
        output = model(data)
        loss = criterion(output, target)
        # just make the loss 0 on bankrupt examples so that we don't have to backprop or clip norms
        # loss[violations] = 0
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.step()
        lr = args.lr
        if scheduler is not None: # doesn't matter since we don't use an lr scheduler but if we do then we need to update it correctly
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        if args.do_budget:
            # do some logging 
            violations = privacy_engine.accountant.get_violations(indices) # we need to know what to tell our optimizer to ignore
            real_step = optimizer.step(violations) # modified to return a boolean; can be false if we are doing gradient accumulation
            privacy_engine.accountant.compute_norms(indices) # update cache with norms now that backprop is done
            if real_step:
                # if we are done accumulating gradients and we are going to do an actual step of optimization
                ema.update() # update our EMA
                privacy_engine.accountant.step()
                wandb.log({"loss": losses.val,
                "top1": top1.val,
                "lr": lr,
                "violations": torch.sum(privacy_engine.accountant.violations),
                "privacy": torch.mean(privacy_engine.accountant.privacy_usage)})
        else:
            optimizer.step()
            ema.update() # update our EMA
            wandb.log({"loss": losses.val,
                "top1": top1.val,
                "lr": lr,
                "privacy": privacy_engine.accountant.get_epsilon(delta=1e-5)})

        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if rank == 0:
            if i % int(args.print_freq * (args.batch_size // max_physical_batch_size)) == 0:
                # if not args.disable_dp:
                #     epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                #         delta=args.delta,
                #         alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                #     )
                #     print(f'Epoch: [{epoch}][{i}/{int(len(train_loader) * args.batch_size/max_physical_batch_size)}]\t'
                #         f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                #         f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                #         f'(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}'
                #     )
                # else:
                print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Lr {lr:.4f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1, lr=lr))
                if args.do_budget:
                    privacy_engine.accountant.print_stats()
                    # stop_training = privacy_engine.accountant.print_stats()
                    # if stop_training:
                    #     print("TOO MANY VIOLATIONS, DONE TRAINING")
                    #     privacy_engine.accountant.cleanup()
                    #     sys.exit()
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, ema, criterion, epoch, privacy_engine, rank=0):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_ema = AverageMeter()
    top1_ema = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ema.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        # target = target.cuda(non_blocking=True)
        # data = data.cuda(non_blocking=True)
        data, target = data.to(rank), target.to(rank)

        # compute output
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        
        # compute output
        with torch.no_grad():
            output = ema(data)
        loss_ema = criterion(output, target)
        
        # and for ema as well
        prec1_ema = accuracy(output.data, target, topk=(1,))[0]
        losses_ema.update(loss_ema.data.item(), data.size(0))
        top1_ema.update(prec1_ema.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Prec@1 (EMA) {top1.avg:.3f}'.format(top1=top1_ema))
    if args.do_budget:
        privacy_engine.accountant.print_stats()
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return max(top1.avg, top1_ema.avg)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')




if __name__ == '__main__':
    main()