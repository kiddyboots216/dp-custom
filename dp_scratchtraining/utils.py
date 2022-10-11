import os
import argparse
import time
import torch
from datetime import datetime
import ctypes
import numpy as np
from collections import namedtuple

import torchvision

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset (cifar10 [default] or cifar100)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run (default: 50)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='mini-batch size (default: 1024)')
    parser.add_argument('--lr', '--learning-rate', default=1.0, type=float,
                        help='initial learning rate (default: 1)')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum (default: 0)')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                        help='weight decay (default: 0)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--layers', default=16, type=int,
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen_factor', default=4, type=int,
                        help='widen factor (default: 10)')
    parser.add_argument('--droprate', default=0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--do_augment', action='store_true',
                        help='whether to use standard augmentation (default: False)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='DP', type=str,
                        help='name of experiment')
    parser.add_argument('--tensorboard',
                        help='Log progress to TensorBoard', action='store_true')
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.5)",
    ) 
    parser.add_argument(
        "-c",
        "--max_per_sample_grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 10.0)",
    )
    parser.add_argument(
        "--disable_dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure_rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommender to turn it on.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2,
        metavar="E",
        help="Target epsilon (default: 2)",
    )
    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )
    # new args
    parser.add_argument(
        "--do_topk",
        action="store_false",
        default=True,
        help="do topk"
    )
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Percentage of params to update"
    )
    parser.add_argument(
        "--do_induced",
        action="store_false",
        default=True,
        help="do the induced compression method from Horvath 2021"
    )
    parser.add_argument(
        "--port",
        type=str,
        help="provide a port so multiple runs don't use same TCP port"
    )
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--do_baseline",
        action="store_false",
        default=True,
        help="do the non-private baseline (alg 3) or something else"
    )
    parser.add_argument(
        "--augmult",
        type=int,
        default=16,
        help="number of augmult augmentations (default: 16)"
    )
    parser.add_argument(
        "--do_augmult",
        action="store_false",
        default=True,
        help="do augmult strategy of multiple data augmentations (default: True)"
    )
    parser.add_argument(
        "--n_stale",
        type=int,
        default=0,
        help="number of iterations to continue using a stale top-k mask"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of dataloader workers"
    )
    parser.add_argument(
        "--do_budget",
        action="store_false",
        default=True,
        help="use influence accounting budgeting strategy"
    )
    # parser.add_argument(
    #     "--influence-budget",
    #     type=float,
    #     default=1.0,
    #     help="maximum influence of each example",
    # )
    
    
    args = parser.parse_args()
    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
    assert not(args.do_augment and args.do_augmult), "You can't do standard data augmentation AND data augmultation, pick one"
    return args

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })