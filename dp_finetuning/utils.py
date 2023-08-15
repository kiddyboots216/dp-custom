import os
import argparse
from collections import defaultdict

import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)


DATASET_TO_CLASSES = {
    "CelebA": -1,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "EMNIST": 62,
    "FashionMNIST": 10,
    "MNIST": 10,
    "STL10": 10,
    "SVHN": 10,
    "waterbirds": 2,
    "fmow": 62,
    "camelyon17": 2,
    "iwildcam": 186,
    "domainnet": 345,
    "ImageNet": 1000,
}
TRANSFER_DATASETS_TO_CLASSES = {
    "STL10_CIFAR": 10,
    "CIFAR10p1": 10,
    "CIFAR10": 10,
    "STL10": 10,
    "CIFAR10C": 10,
    "CIFAR100C": 100,
}
DATASET_TO_SIZE = {
    'CIFAR10': 50000,
    'CIFAR100': 50000,  
    'STL10': 5000,
    'FashionMNIST': 60000,
    'EMNIST': 697932,
    'waterbirds': 4795,
    'domainnet': 48212,
    'fmow': 20973,
    "ImageNet": 1281167,
 }
ARCH_TO_NUM_FEATURES = {
    "beitv2_large_patch16_224_in22k": 1024,
    "beitv2_large_patch16_224": 1024,
    "beit_large_patch16_512": 1024,
    "convnext_xlarge_384_in22ft1k": 2048,
    "vit_large_patch16_384": 1024,
    "vit_gigantic_patch14_clip_224.laion2b": 1664,
    "vit_giant_patch14_224_clip_laion2b": 1408,
    "stylegan-resnet50-bn": 2048,
    "shaders-resnet50-bn": 2048,
    "stylegan-resnet50-gn": 2048,
    "stylegan-nfnet": 2048,
    'stylegan-wideresnet16': 256,
    "resnet20": 1024,
    "vit_base_patch16": {"": 768 * 12,
                         "fixed": 768 * 12 * 7,
                         "avgpool": 768 * 12 * 4},
}
ARCH_TO_INTERP_SIZE = {
    "beitv2_large_patch16_224_in22k": 224,
    "beitv2_large_patch16_224": 224,
    "beit_large_patch16_512": 512,
    "convnext_xlarge_384_in22ft1k": 384,
    "vit_large_patch16_384": 384,
    "vit_base_patch16_384": 384,
    "tf_efficientnet_l2_ns": 800,
    "vit_gigantic_patch14_clip_224.laion2b": 224,
    "vit_giant_patch14_224_clip_laion2b": 224,
    "stylegan-resnet50-bn": 224,
    "shaders-resnet50-bn": 224,
    "stylegan-resnet50-gn": 224,
    "stylegan-nfnet": 224,
    'stylegan-wideresnet16': 224,
    "vit_base_patch16": 224,
    "resnet20": 224,
}

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DP Finetuning")
    parser.add_argument(
        "--dataset",
        default="CIFAR10",
        type=str,
        choices=list(DATASET_TO_CLASSES.keys()),
    )
    parser.add_argument(
        "--transfer_dataset",
        default="STL10_CIFAR",
        type=str,
        choices=list(TRANSFER_DATASETS_TO_CLASSES.keys()),
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=list(ARCH_TO_NUM_FEATURES.keys()),
        default=list(ARCH_TO_NUM_FEATURES.keys())[0],
    )
    parser.add_argument("--lr", default=1, type=float)
    parser.add_argument("--epochs", default=110, type=int)
    parser.add_argument("--batch_size", default=-1, type=int)
    parser.add_argument("--sigma", default=37.80, type=float, help="The default sigma is for epsilon=1.0, passing sigma=0.0 will disable privacy, and sigma=-1 will calculate sigma from epsilon")
    parser.add_argument(
        "--max_per_sample_grad_norm",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--disable_dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--standardize_weights",
        action="store_false",
        default=True,
        help="Initialize weights to zero to make the initialization better",
    )
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument(
        "--secure_rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommended to turn it on.",
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
        default=1,
        metavar="E",
        help="Target epsilon (default: 1)",
    )
    parser.add_argument(
        "--feature_norm",
        type=float,
        default=100,
        help="Target feature norm (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11297,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="extracted_features/",
    )
    parser.add_argument(
        "--augmult",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sched", choices=['0', '1', '2'], default='0', help="Use learning rate schedule"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "individual",
            "dpsgdfilter",
            "vanilla",
            "sampling",
        ],
        default="vanilla",
        help="What mode of DPSGD optimization to use. Individual and Dpsgdfilter both use GDP filter.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="How many runs to generate "
    )
    parser.add_argument(
        "--store_grads",
        action="store_true",
        default=False,
        help="Store gradients for each epoch",
    )
    parser.add_argument(
        "--store_weights",
        action="store_true",
        default=False,
        help="Store only the final weights",
    )
    parser.add_argument(
        "--max_phys_bsz",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--optimizer",
        choices=[
            "sgd",
            "adam",
            "lamb",
            "pgd",
        ],
        default="sgd",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files with new ones",
    )
    parser.add_argument(
        "--feature_mod",
        type=str,
        default="",
        help="If specified with --overwrite, overwritten features will be written to this path. Otherwise, features will be loaded from the path ending with feature_mod, e.g., train{feature_mod}.npy"
    )
    parser.add_argument(
        "--privacy_engine",
        type=str,
        default="fastDP",
        choices=["fastDP", "opacus"],
        help="what privacy engine to use"
    )
    # parser.add_argument(
    #     "--weight_avg_mode",
    #     choices=[
    #         "none",
    #         "ema",
    #         "swa",
    #         "best",
    #     ],
    #     default="best",
    #     help="What kind of weight averaging to use",
    # )
    args = parser.parse_args()
    if args.batch_size == -1:
        args.batch_size = DATASET_TO_SIZE[args.dataset]
    args.num_classes = DATASET_TO_CLASSES[args.dataset]
    if args.epsilon == 0.0:
        args.sigma = 0
    if args.sigma == -1:
        args.delta = 1 / (2 * DATASET_TO_SIZE[args.dataset])
        # let the prv acct determine sigma for us
        from prv_accountant.dpsgd import find_noise_multiplier

        sampling_probability = args.batch_size / DATASET_TO_SIZE[args.dataset]
        args.sigma = find_noise_multiplier(
            sampling_probability=sampling_probability,
            num_steps=int(args.epochs / sampling_probability),
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            eps_error=0.01,
            mu_max=5000)
        print("Using noise multiplier", args.sigma)
    for arg in vars(args):
        print(" {} {}".format(arg, getattr(args, arg) or ""))
    return args
import torchvision
def create_resnet20_from_pretrained(args):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
    from collections import OrderedDict

    def modify_model_correctly(model):
        # Convert the model's children into a list
        children_list = list(model.children())
        
        # Modify the second to last layer and exclude the last layer
        modified_children = children_list[:-2] + [nn.AdaptiveAvgPool2d((4, 4))]
        
        # Convert the modified children list back to a Sequential model
        modified_model = nn.Sequential(*modified_children)
        return modified_model


    # Usage:
    modified_model = modify_model_correctly(model)
    return modified_model


def create_model_from_pretrained(args):
    model = getattr(torchvision.models, 'resnet50')(weights=None)
    if args.arch == 'stylegan-resnet50-bn':
        top_pth = '/scratch/gpfs/USER/learning_with_noise/scripts/download_pretrained_models/encoders/large_scale/stylegan-oriented'
        ckpt_path = 'checkpoint_0199.pth.tar'
    elif args.arch == 'shaders-resnet50-bn':
        top_pth = '/scratch/gpfs/USER/shaders21k/scripts/download/encoders/shaders21k_6mixup'
        ckpt_path = 'checkpoint_0199.pth.tar'
    elif args.arch == 'stylegan-resnet50-gn':
        top_pth = '/scratch/gpfs/USER/learning_with_noise/encoders/large_scale/stylegan-oriented-resnet50/'
        ckpt_path = 'checkpoint_0156.pth.tar'
    elif args.arch == 'stylegan-wideresnet16':
        top_pth = '/scratch/gpfs/USER/learning_with_noise/encoders/large_scale/stylegan-oriented-wideresnet16/'
        ckpt_path = 'encoder.pth'
    else:
        # raise NotImplementedError
        raise NotImplementedError(f"Unknown arch: {args.arch}")
    encoder_checkpoint = os.path.join(top_pth, ckpt_path)
    # Load the checkpoint
    print(f"Loading from: {encoder_checkpoint}")
    ckpt = torch.load(encoder_checkpoint, map_location='cpu')

    # rename moco_training pre-trained keys
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    
    if 'gn' in args.arch: # torchvision rn50 has bn, we need to replace these with gn
        # Replace all the Layers we don't like with Layers that we do like
        def replace_layers_rn(model):
            for name, module in reversed(list(model._modules.items())):
                if len(list(module.children())) > 0:
                    model._modules[name] = replace_layers_rn(module)
                if isinstance(module, nn.BatchNorm2d):
                    num_channels = module.num_features
                    num_groups = max(num_channels // 32, 1)
                    model._modules[name] = nn.GroupNorm(num_groups, num_channels)
            return model

        model = replace_layers_rn(model)
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, msg
    # remove the fc layer because we are just extracting features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    # Freeze all the layers except the last one
    for param in model.parameters():
        param.requires_grad = False

    return model
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(
        cls.__name__,
        (cls,),
        {
            "__getitem__": __getitem__,
        },
    )
def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

def get_features(f, images, interp_size=224, batch=64):
    features = []
    # hardcode to 336
    for img in tqdm(images.split(batch)):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # for img in tqdm(images):
            img = F.interpolate(
                img.cuda(), size=(interp_size, interp_size), mode="bicubic"
            )  # up-res size hardcoded
            features.append(f(img).detach().cpu())
    return torch.cat(features)

def extract_features_from_model(model, dl, args, max_count=10**9):
    feat, labels = [], []
    count = 0
    for img, label in tqdm(dl):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # out = model(img.cuda())
            if args.feature_mod == "avgpool":
                feat.append(model.forward_features_avgpool(img.cuda()).detach().cpu().float())
            elif args.feature_mod == "fixed":
                feat.append(model.forward_features_dense(img.cuda()).detach().cpu().float())
            else:
                feat.append(model.forward_features(img.cuda()).detach().cpu().float())
        labels.append(label)
        count += len(img)
        if count >= max_count:
            break
    return torch.cat(feat), torch.cat(labels)

def download_things(args):
    dataset_path = args.dataset_path
    if args.dataset in ["SVHN", "STL10"]:
        ds = getattr(datasets, args.dataset)(
            dataset_path, transform=transforms.ToTensor(), split="train", download=True
        )
    elif args.dataset in ["EMNIST"]:
        ds = getattr(datasets, args.dataset)(
            dataset_path,
            transform=transforms.ToTensor(),
            split="byclass",
            download=True,
        )
    else:
        ds = getattr(datasets, args.dataset)(
            dataset_path, transform=transforms.ToTensor(), train=True, download=True
        )
    feature_extractor = nn.DataParallel(
        timm.create_model(args.arch, num_classes=0, pretrained=True)
    )

def get_extract_features_from_model(model, dl, args, max_count=10**9):
    feat, labels = [], []
    count = 0
    model = model.cuda()
    for img, label in tqdm(dl):
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model(img.cuda())
            # print(out.shape)
        if 'resnet' in args.arch:
            feat.append(out.detach().cpu().float().reshape(out.shape[0], -1)) # do this for rn50
        elif 'nfnet' in args.arch:
            feat.append(out.detach().cpu().float()) # do this for timm models aka nfnet
        labels.append(label)
        count += len(img)
        if count >= max_count:
            break
    return torch.cat(feat), torch.cat(labels)

def extract_features(args, images_train=None, images_test=None):
    ### GET PATH
    abbrev_arch = args.arch
    extracted_path = (
        args.dataset_path
        + "transfer/features/"
        + args.dataset.lower()
        + "_"
        + abbrev_arch
    )
    extracted_train_path = extracted_path + "/_train.npy"
    extracted_test_path = extracted_path + "/_test.npy"

    ### DO EXTRACTION

    print("GENERATING AND SAVING EXTRACTED FEATURES AT ", extracted_train_path)
    feature_extractor = (
        nn.DataParallel(timm.create_model(args.arch, num_classes=0, pretrained=True))
        .eval()
        .cuda()
    )
    interp_size = ARCH_TO_INTERP_SIZE[args.arch]
    batch_extract_size = 128

    if images_train is not None:
        features_train = get_features(
            feature_extractor,
            images_train,
            interp_size=interp_size,
            batch=batch_extract_size,
        )
        os.makedirs(extracted_path, exist_ok=True)
        np.save(extracted_train_path, features_train)
    if images_test is not None:
        features_test = get_features(
            feature_extractor,
            images_test,
            interp_size=interp_size,
            batch=batch_extract_size,
        )
        os.makedirs(extracted_path, exist_ok=True)
        np.save(extracted_test_path, features_test)

def check_zero_ones_range_tensor(x):
    assert torch.ge(x, 0).prod() and torch.le(x, 1).prod(
    ), f"Input must be in [0, 1] range. Current range: {[x.min().item(), x.max().item()]}"

class timmFe(nn.Module):
    """
    This class is a wrapper around timm models to use them as feature extractors. 
    We automatically resize images, normalize them and remove the last layer of model to extract features.
    """
    def __init__(self, arch, ckpt_path=''):
        super(timmFe, self).__init__()
        self.net = timm.create_model(arch, pretrained=True, num_classes=0)
        self.h, self.w = self.net.default_cfg['input_size'][1:]
        if ckpt_path:
            print("Loading feature extractor checkpoint")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print("Following keys are not available in feat-extractor ckpt: ", 
                    set(ckpt.keys()) ^ set(self.net.state_dict().keys()))
            self.net.load_state_dict(ckpt, strict=False)
            print(f"Feat extractor successfully loaded from: {ckpt_path}")
        self.register_buffer('mean', torch.tensor(self.net.default_cfg['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.net.default_cfg['mean']).view(1, 3, 1, 1))
        self.normalize_input = True
    def forward(self, x):
        check_zero_ones_range_tensor(x)
        if x.shape[-2:] != torch.Size([self.h, self.w]):
            x = F.interpolate(x, size=(self.h, self.w), mode='bicubic', align_corners=False)
            x = x.clamp(0, 1)
        if self.normalize_input:
            x = (x - self.mean) / self.std
        return self.net(x)

# # def get_standardized_weight(weight,out_channels,eps,gain, scale):
# #     weight = F.batch_norm(
# #             weight.reshape(1, out_channels, -1), None, None,
# #             weight=(gain * scale).view(-1),
# #             training=True, momentum=0., eps=eps).reshape_as(weight)
# #     return weight

# import numpy as np
# import timm
# from typing import List, Tuple, Dict, Any, Optional
# from timm.layers.padding import get_padding,  get_padding_value, pad_same
# from timm.layers.std_conv import ScaledStdConv2d, ScaledStdConv2dSame
# from timm.models.layers import make_divisible, DropPath
# from timm.models.nfnet import DownsampleAvg, NormFreeBlock
# from opacus.grad_sample import register_grad_sampler

# def get_standardized_weight(weight, out_channels, eps, gain, scale):
#     original_shape = weight.shape
#     weight = weight.reshape(1, out_channels, -1)
    
#     # calculate mean and variance
#     mean = weight.mean(dim=2, keepdim=True)
#     var = weight.var(dim=2, keepdim=True)

#     # standardize the weights (Batch normalization without running averages)
#     weight = (weight - mean) / torch.sqrt(var + eps)

#     # reshape weights to their original shape
#     weight = weight.reshape(original_shape)

#     # apply channel-wise scaling (equivalent to the 'weight' parameter in batch normalization)
#     weight = weight * gain * scale
    
#     return weight



# # def get_standardized_weight(weight, gain=None, eps=1e-4):
# #     # Get Scaled WS weight OIHW;
# #     fan_in = np.prod(weight.shape[-3:])
# #     mean = torch.mean(weight, axis=[-3, -2, -1], keepdims=True)
# #     var = torch.var(weight, axis=[-3, -2, -1], keepdims=True)
# #     weight = (weight - mean) / (var * fan_in + eps) ** 0.5
# #     if gain is not None:
# #         weight = weight * gain
# #     return weight

# class MyScaledStdConv2d(nn.Conv2d):
#     """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

#     Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
#         https://arxiv.org/abs/1903.10520v2
#     """
#     def __init__(
#             self, in_channels, out_channels, kernel_size, stride=1, padding=None,
#             dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
#         if padding is None:
#             padding = get_padding(kernel_size, stride, dilation)
#         super().__init__(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
#             groups=groups, bias=bias)
#         self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
#         self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
#         self.eps = eps
        
#     def forward(self, x):
#         # std_weight = get_standardized_weight(self.weight, gain=self.gain, eps=self.eps)
#         std_weight = get_standardized_weight(self.weight, self.out_channels, self.eps, self.gain, self.scale)
#         return F.conv2d(x, std_weight, self.bias,self.stride, self.padding, self.dilation, self.groups)

# def replace_layers(model):
#     for name, module in reversed(list(model._modules.items())):
#         if len(list(module.children())) > 0:
#             model._modules[name] = replace_layers(module)
#         if isinstance(module, ScaledStdConv2d):
#             # create a MyScaledStdConv2d with the same parameters
#             # copy all the weights over
#             in_channels = module.in_channels
#             out_channels = module.out_channels
#             kernel_size = module.kernel_size
#             stride = module.stride
#             padding = module.padding
#             dilation = module.dilation
#             groups = module.groups
#             bias = module.bias is not None

#             new_module = MyScaledStdConv2d(
#                 in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
#                 dilation=dilation, groups=groups, bias=bias
#             )

#             # Copy weights and biases
#             new_module.weight.data.copy_(model._modules[name].weight.data)
#             if module.bias is not None:
#                 new_module.bias.data.copy_(model._modules[name].bias.data)
#             # copy over the 'gain' parameter
#             new_module.gain.data.copy_(model._modules[name].gain.data)
#             # copy over the 'eps' parameter
#             new_module.eps = model._modules[name].eps
#             # copy over the 'scale' parameter
#             new_module.scale = model._modules[name].scale
#             model._modules[name] = new_module

#     return model 

# # def load_wrn(args):
# #     from wrn import WideResNet 
# #     model = WideResNet(16, 256, 4)

# def load_nfnet(args):
#     # load the nfnet50 model from timm
#     model = timm.create_model('nf_resnet50', num_classes=0)
#     pretrained_path = 'checkpoint_0120.pth.tar'
#     encoder_checkpoint = os.path.join(shaders_path, pretrained_path)

#     # Load the checkpoint
#     print(f"Loading from: {encoder_checkpoint}")
#     ckpt = torch.load(encoder_checkpoint, map_location='cpu')

#     # rename moco_training pre-trained keys
#     state_dict = ckpt['state_dict']
#     for k in list(state_dict.keys()):
#         # retain only encoder_q up to before the embedding layer
#         if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
#             # remove prefix
#             state_dict[k[len("module.encoder_q."):]] = state_dict[k]
#         # delete renamed or unused k
#         del state_dict[k]

#     # load the timm model with the checkpoint
#     msg = model.load_state_dict(state_dict, strict=False)
#     # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, msg.missing_keys
#     model = replace_layers(model)
#     # remove the fc layer because we are just extracting features
#     # model = torch.nn.Sequential(*(list(model.children())[:-1]))
#     model.eval()

#     for param in model.parameters():
#         param.requires_grad = False

#     return model

def load_pretrained_vit(args):
    import util.misc as misc
    from util.pos_embed import interpolate_pos_embed
    
    global_pool = True
    import models_vit
    model = models_vit.__dict__[args.arch](
        num_classes=1000,
        global_pool=global_pool,
        lp_num_layers=12,
    )
    finetune_path = "path/to/vit/checkpoint"
    checkpoint = torch.load(finetune_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % finetune_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    return model

def get_ds(args):
    dataset_path = args.dataset_path
    extracted_path = (
        args.dataset_path
        + "transfer/features/"
        + args.dataset.lower()
        + "_"
        + args.arch
    )
    extracted_train_path = extracted_path + "/_train.npy"
    extracted_test_path = extracted_path + "/_test.npy"
    # fe = timmFe(args.arch).cuda() # timm models based feature extractor
    
    kwargs = {"num_workers": args.workers, "pin_memory": True}
    if not os.path.exists(dataset_path):
        raise Exception(
            "We cannot download a dataset/model here \n Run python utils.py to download things"
        )
    if args.dataset in ["ImageNet"]:
        # most models finally require squared images, so h and w are equal
        # assert ARCH_TO_INTERP_SIZE[args.arch] == fe.h, f"Interpolation size {ARCH_TO_INTERP_SIZE[args.arch]} is not equal to {fe.h} h of the model"
        def get_feature_path(base_path, feature_mod, default_suffix=".npy"):
            if feature_mod:
                return base_path.replace(default_suffix, f"{feature_mod}.npy")
            return base_path

        # Determine paths for extracted features and labels
        extracted_train_path = get_feature_path(extracted_path + "/_train.npy", args.feature_mod)
        labels_train_path = extracted_path + "/_train_labels.npy"
        extracted_test_path = get_feature_path(extracted_path + "/_test.npy", args.feature_mod)
        labels_test_path = extracted_path + "/_test_labels.npy"

        # Handle overwriting logic
        if args.overwrite:
            print(f"Overwriting existing feature file for {extracted_train_path}...")
            if not args.feature_mod:  # If no custom modification is provided, use 'tmp' as default
                extracted_train_path = get_feature_path(extracted_train_path, "_tmp")
                extracted_test_path = get_feature_path(extracted_test_path, "_tmp")
        
        if os.path.exists(extracted_train_path) and not args.overwrite:
            print(f"Features already extracted for {extracted_train_path}, skipping...")
        else:
            print(f"Extracting features to {extracted_train_path}...")
            if "vit" in args.arch:
                model = load_pretrained_vit(args) 
            elif "nfnet" in args.arch:
                # model = timm.create_model(args.arch, pretrained=True, num_classes=0)
                model = load_nfnet(args)
            elif "resnet" in args.arch:
                model = create_model_from_pretrained(args)
            model.eval()
            model = model.cuda()
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
            IMAGENET_STD = np.array([0.229, 0.224, 0.225])
            from util.crop import RandomResizedCrop
            transform_train = transforms.Compose([
                RandomResizedCrop(224, interpolation=3),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize(256, interpolation=3),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
            transform_val = transforms.Compose([
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])      
            transform_train = transform_val # just testing
            dataset_path = args.dataset_path + args.dataset
            train_path = dataset_path + "/train"
            train_ds = datasets.ImageFolder(train_path, transform=transform_train)
            batch_extract_size = 1024
            train_loader = DataLoader(train_ds, batch_size=batch_extract_size, shuffle=False, num_workers=args.workers, pin_memory=False)
            features_train, labels_train = extract_features_from_model(model, train_loader, args)
            os.makedirs(extracted_path, exist_ok=True)
            np.save(extracted_train_path, features_train)
            np.save(labels_train_path, labels_train)
            print("Extracted features from train set")
            test_path = dataset_path + "/val"
            test_ds = datasets.ImageFolder(test_path, transform=transform_val)
            test_loader = DataLoader(test_ds, batch_size=batch_extract_size, shuffle=False, num_workers=args.workers, pin_memory=False)
            features_test, labels_test = extract_features_from_model(model, test_loader, args)
            np.save(extracted_test_path, features_test)
            np.save(labels_test_path, labels_test)
            print("Extracted features from test set")
        # features_train = torch.load(extracted_train_path)
        # features_test = torch.load(extracted_test_path)
        # labels_train = torch.load(labels_train_path)
        # labels_test = torch.load(labels_test_path)
        # if True:
        #     extracted_train_path = extracted_train_path.replace(".npy", "tmp.npy")
        #     extracted_test_path = extracted_test_path.replace(".npy", "tmp.npy")
        #     print("Using overwritten features from disk")
        print("Loading features from disk")
        features_train = torch.from_numpy(np.load(extracted_train_path))
        features_test = torch.from_numpy(np.load(extracted_test_path))

        print("Loading labels from disk")
        labels_train = torch.from_numpy(np.load(labels_train_path))
        labels_test = torch.from_numpy(np.load(labels_test_path))

        # def preprocess_features_tensor(features_tensor, desired_norm=args.feature_norm):
        #     split_tensors = torch.split(features_tensor, 768, dim=1)  # Split into 12 chunks of 768 dims each
            
        #     normalized_splits = []
        #     for tensor_chunk in split_tensors:
        #         norms = torch.norm(tensor_chunk, dim=1, keepdim=True)
                
        #         # If desired_norm is -1, normalize to the mean norm of the chunk
        #         target_norm = norms.mean() if desired_norm == -1 else desired_norm
        #         normalized_chunk = (tensor_chunk / norms) * target_norm
                
        #         mean_vector = torch.mean(normalized_chunk, dim=0)
        #         translated_chunk = normalized_chunk - mean_vector
                
        #         normalized_splits.append(translated_chunk)
            
        #     # Concatenate all the normalized chunks back together
        #     return torch.cat(normalized_splits, dim=1)
        
        def preprocess_features_tensor(features_tensor, desired_norm=args.feature_norm):
            # Normalize feature vectors
            norms = torch.norm(features_tensor, dim=1, keepdim=True)
            target_norm = norms.mean() if desired_norm == -1 else desired_norm
            normalized_features = (features_tensor / norms) * target_norm
            
            # Compute the mean of the normalized features
            mean_vector = torch.mean(normalized_features, dim=0)
            
            # Subtract the mean from the normalized features
            translated_features = normalized_features - mean_vector
            
            return translated_features

        if args.feature_norm != 0:  # Adjusted condition to cater for -1 as a valid input
            features_train = preprocess_features_tensor(features_train, desired_norm=args.feature_norm)
            features_test = preprocess_features_tensor(features_test, desired_norm=args.feature_norm)


        ds_train = dataset_with_indices(TensorDataset)(features_train, labels_train)
        ds_test = TensorDataset(features_test, labels_test)
        train_loader = DataLoader(
            ds_train, batch_size=args.max_phys_bsz, shuffle=True, **kwargs
        )
        test_loader = DataLoader(
            ds_test, batch_size=args.max_phys_bsz, shuffle=False, **kwargs
        )
        return train_loader, test_loader
    elif args.dataset in ["STL10_CIFAR"]:
        print("Loading STL10_CIFAR")
        from stl_cifar_style import STL10 as STL10_CIFAR
        STL_CIFAR_dataset = STL10_CIFAR(
            root="/data/nvme/$USER/datasets",
            split="test",
            folds=None,
            transform=None,
            target_transform=None,
            download=False,
        )
        images_test = torch.tensor(STL_CIFAR_dataset.data) / 255.0
        labels_test = torch.tensor(STL_CIFAR_dataset.labels)
        if not os.path.exists(extracted_path):
            extract_features(
                args, images_train=None, images_test=images_test
            )  # don't want to extract train features
        x_test = np.load(extracted_test_path)
        features_test = torch.from_numpy(x_test)
        ds_test = TensorDataset(features_test, labels_test)
        test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
        return None, test_loader, None, None  # no train loader
    elif args.dataset in ["CIFAR10p1"]:
        print("Loading CIFAR10p1")
        from cifar10p1 import CIFAR10p1
        CIFAR10p1_dataset = CIFAR10p1(
            root="/home/$USER/CIFAR-10.1/datasets/",  # download from https://github.com/modestyachts/CIFAR-10.1
            split="test",
            transform=None,
        )
        images_test = (
            torch.tensor(CIFAR10p1_dataset._imagedata.transpose(0, 3, 1, 2)) / 255.0
        )
        labels_test = torch.tensor(CIFAR10p1_dataset._labels)
        if not os.path.exists(extracted_path):
            extract_features(
                args, images_train=None, images_test=images_test
            )  # don't want to extract train features
        x_test = np.load(extracted_test_path)
        features_test = torch.from_numpy(x_test)
        ds_test = TensorDataset(features_test, labels_test)
        test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
        return None, test_loader, None, None  # no train loader
    elif args.dataset in ["CIFAR10C"]:
        print("Loading CIFAR10C")
        from cifar10c import CIFAR10C
        CIFAR10C_dataset = CIFAR10C(
            root="/data/$USER/datasets/CIFAR10C",
            corruption="gaussian_noise",
            severity=2,
            transform=None,
        )
        images_test = torch.tensor(CIFAR10C_dataset._xs.transpose(0, 3, 1, 2)) / 255.0
        labels_test = CIFAR10C_dataset._ys
        if not os.path.exists(extracted_path):
            extract_features(args, images_train=None, images_test=images_test)
        x_test = np.load(extracted_test_path)
        features_test = torch.from_numpy(x_test)
        ds_test = TensorDataset(features_test, labels_test)
        test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
        return None, test_loader, None, None  # no train loader
    elif args.dataset in ["CIFAR100C"]:
        print("Loading CIFAR100C")
        CIFAR100C_dataset = CIFAR10C(
            root="/data/$USER/datasets/CIFAR100C",
            corruption="gaussian_noise",
            severity=2,
            transform=None,
        )
        images_test = torch.tensor(CIFAR100C_dataset._xs.transpose(0, 3, 1, 2)) / 255.0
        labels_test = CIFAR100C_dataset._ys
        if not os.path.exists(extracted_path):
            extract_features(args, images_train=None, images_test=images_test)
        x_test = np.load(extracted_test_path)
        features_test = torch.from_numpy(x_test)
        ds_test = TensorDataset(features_test, labels_test)
        test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
        return None, test_loader, None, None  # no train loader
    elif args.dataset in ["SVHN", "STL10"]:
        ds = getattr(datasets, args.dataset)(
            dataset_path, transform=transforms.ToTensor(), split="train"
        )
        images_train, labels_train = torch.tensor(ds.data) / 255.0, torch.tensor(
            ds.labels
        )
        ds = getattr(datasets, args.dataset)(
            dataset_path, transform=transforms.ToTensor(), split="test"
        )
        images_test, labels_test = torch.tensor(ds.data) / 255.0, torch.tensor(
            ds.labels
        )
    elif "MNIST" in args.dataset:
        if args.dataset == "EMNIST":
            ds_train = getattr(datasets, args.dataset)(
                dataset_path,
                transform=transforms.ToTensor(),
                split="byclass",
                train=True,
            )
            ds_test = getattr(datasets, args.dataset)(
                dataset_path,
                transform=transforms.ToTensor(),
                split="byclass",
                train=False,
            )
        else:
            ds_train = getattr(datasets, args.dataset)(
                dataset_path, transform=transforms.ToTensor(), train=True
            )
            ds_test = getattr(datasets, args.dataset)(
                dataset_path, transform=transforms.ToTensor(), train=False
            )
        print("MAKING IMAGES_TRAIN")
        labels_train = torch.tensor(ds_train.targets)
        labels_test = torch.tensor(ds_test.targets)
        images_train, labels_train = torch.tensor(
            ds_train.data.unsqueeze(1).repeat(1, 3, 1, 1)
        ).float() / 255.0, torch.tensor(ds_train.targets)
        images_test, labels_test = torch.tensor(
            ds_test.data.unsqueeze(1).repeat(1, 3, 1, 1)
        ).float() / 255.0, torch.tensor(ds_test.targets)
    elif "CIFAR100" in args.dataset:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.ColorJitter(brightness=(0.2), saturation=(0.2), hue=(0.2)),
            ]
        )
        train_ds = getattr(datasets, args.dataset)(
            dataset_path, transform=transform, train=True
        )
        images_train, labels_train = torch.tensor(
            train_ds.data.transpose(0, 3, 1, 2)
        ) / 255.0, torch.tensor(train_ds.targets)
        test_ds = getattr(datasets, args.dataset)(
            dataset_path, transform=transform, train=False
        )
        images_test, labels_test = torch.tensor(
            test_ds.data.transpose(0, 3, 1, 2)
        ) / 255.0, torch.tensor(test_ds.targets)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)
    else:
        extracted_train_path_part_1 = extracted_path + "/part1_train.npy"
        extracted_train_path_part_2 = extracted_path + "/part2_train.npy"
        labels_train_path = extracted_path + "/_train_labels.npy"
        extracted_test_path = extracted_path + "/_test.npy"
        labels_test_path = extracted_path + "/_test_labels.npy"
        if not os.path.exists(extracted_train_path_part_1):
        # if True:
            # model = create_resnet20_from_pretrained(args)
                    # CIFAR_MEAN = np.array((0.4914, 0.4822, 0.4465))
            CIFAR_STD = np.array((0.247, 0.243, 0.261))
            train_tr = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.ToTensor(),
                    # normalize ImageNet according to its mean and standard deviation
                    # transforms.Normalize(  
                    #     mean=CIFAR_MEAN, std=CIFAR_STD
                    # ),
                ])      
            ds = getattr(datasets, args.dataset)(
                dataset_path, transform=train_tr, train=True
            )
            images_train, labels_train = torch.tensor(
                ds.data.transpose(0, 3, 1, 2)
            ) / 255.0, torch.tensor(ds.targets)
            test_tr = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(  
                #         mean=CIFAR_MEAN, std=CIFAR_STD
                #     ),
            ])
            ds = getattr(datasets, args.dataset)(
                dataset_path, transform=test_tr, train=False
            )
            images_test, labels_test = torch.tensor(
                ds.data.transpose(0, 3, 1, 2)
            ) / 255.0, torch.tensor(ds.targets)
            # model = create_model_from_pretrained(args)
            model = timmFe(args.arch).cuda() # if you get any errors around here it may because I had to fiddle with things to upload the extracted features to GitHub
            dataset_path = args.dataset_path + args.dataset
            batch_extract_size = 1024
            train_ds = TensorDataset(images_train, labels_train)
            test_ds = TensorDataset(images_test, labels_test)
            train_loader = DataLoader(train_ds, batch_size=batch_extract_size, shuffle=False, num_workers=args.workers, pin_memory=False)
            features_train, labels_train = get_extract_features_from_model(model, train_loader, args)
            os.makedirs(extracted_path, exist_ok=True)
            np.save(extracted_train_path, features_train)
            np.save(labels_train_path, labels_train)
            print("Extracted features from train set")
            test_path = dataset_path + "/val"
            test_loader = DataLoader(test_ds, batch_size=batch_extract_size, shuffle=False, num_workers=args.workers, pin_memory=False)
            features_test, labels_test = get_extract_features_from_model(model, test_loader, args)
            np.save(extracted_test_path, features_test)
            np.save(labels_test_path, labels_test)
            print("Extracted features from test set")
    if args.dataset == "CIFAR10":
        # special case because we uploaded these to git and had to split them
        x_train_part_1 = np.load(extracted_train_path_part_1)
        x_train_part_2 = np.load(extracted_train_path_part_2)
        x_train = np.concatenate((x_train_part_1, x_train_part_2), axis=0)
    else:
        if not os.path.exists(extracted_train_path):
            # model = create_model_from_pretrained(args)
            extract_features(args, images_train, images_test)
        x_train = np.load(extracted_train_path)
    features_train = torch.from_numpy(x_train)
    labels_train = torch.from_numpy(np.load(labels_train_path))
    labels_test = torch.from_numpy(np.load(labels_test_path))
    ds_train = dataset_with_indices(TensorDataset)(features_train, labels_train)
    if args.batch_size == -1:
        args.batch_size = len(ds_train)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    x_test = np.load(extracted_test_path)
    features_test = torch.from_numpy(x_test)
    ds_test = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, **kwargs)
    return train_loader, test_loader