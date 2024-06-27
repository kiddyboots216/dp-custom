import os
import argparse
from collections import defaultdict
import glob

import timm
import torch
import numpy as np
from functools import partial
from torch.utils.data import Subset

import torch
import torch.nn as nn

import timm.models.vision_transformer
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
    "vit_gigantic_patch14_clip_224.laion2b": {"none": 1664},
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
    "eva02_large_patch14_448.mim_m38m_ft_in22k": {"base": 98304},
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
    "eva02_large_patch14_448.mim_m38m_ft_in22k": 448,
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
    parser.add_argument("--sigma", default=-1, type=float, help="You can pass sigma=37.80 for epsilon=1.0, passing sigma=0.0 will disable privacy, and sigma=-1 will calculate sigma from epsilon")
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
        "--zo_eps",
        type=float,
        default=1e-3,
        help="Zo eps",
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
        "--start_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
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
        "--save_weights",
        action="store_true",
        default=False,
        help="Store only the final weights",
    )
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
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
        default="none",
        help="If specified with --overwrite, overwritten features will be written to this path. Otherwise, features will be loaded from the path ending with feature_mod, e.g., train{feature_mod}.npy"
    )
    parser.add_argument(
        "--privacy_engine",
        type=str,
        default="fastDP",
        choices=["fastDP", "opacus"],
        help="what privacy engine to use"
    )
    parser.add_argument("--list_lr", default="[1]", type=parse_list)

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
        if args.end_idx == 625:
            sampling_probability *= 2
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
# In your argument parsing script (utils.py)
import ast

def parse_list(arg):
    return ast.literal_eval(arg)
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

def extract_features(args, images_train=None, images_test=None, labels_train=None, labels_test=None):
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
    labels_train_path = extracted_path + "/_train_labels.npy"
    labels_test_path = extracted_path + "/_test_labels.npy"
    np.save(labels_train_path, labels_train)
    np.save(labels_test_path, labels_test)

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

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, num_layers=12, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.fc_norm = kwargs['norm_layer'](kwargs['embed_dim'])
        del self.norm

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        feature_list = []
        idx_block = 0

        for blk in self.blocks:
            x = blk(x)
            idx_block += 1
            if idx_block > (len(self.blocks) - self.num_layers):
                x_no_cls = x[:, 1:, ...]
                
                # Check the dimensionality of x_no_cls
                if len(x_no_cls.shape) == 3:  # shape is [B, T, F]
                    # Calculate the stride for pooling so that the output T dimension is approximately 4.
                    stride = max(1, x_no_cls.shape[1] // 4)
                    pooled_output = F.avg_pool1d(x_no_cls.permute(0, 2, 1), kernel_size=stride, stride=stride).permute(0, 2, 1)
                elif len(x_no_cls.shape) == 4:  # shape is [B, T1, T2, F]
                    # Calculate the stride for pooling so that the output T1 and T2 dimensions are approximately 4.
                    stride = max(1, x_no_cls.shape[2] // 4)
                    pooled_output = F.avg_pool2d(x_no_cls, kernel_size=(stride, stride), stride=stride)
                else:
                    raise ValueError(f"Unexpected tensor shape: {x_no_cls.shape}")

                # Flatten the pooled output across the time dimension
                flattened_output = pooled_output.reshape(pooled_output.size(0), -1)
                
                feature_list.append(flattened_output)


        outcome = torch.cat(feature_list, 1)

        return outcome

    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)


def load_pretrained_vit(args):    
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000,
        num_layers=12,)
    finetune_path = "path/to/vit/checkpoint" # replace this with your path
    finetune_path = "ckpt-vip-syn-base.pth"
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

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    return model

def load_custom_eva(args):

    class CustomModelWrapper:
        def __init__(self, base_model, num_layers=12):
            self.base_model = base_model
            self.num_layers = num_layers

        def forward_features(self, x):
            x = self.base_model.patch_embed(x)
            # Use the _pos_embed function from the base model to get position embeddings and rotary position embeddings
            x, rot_pos_embed = self.base_model._pos_embed(x)

            feature_list = []

            for blk in self.base_model.blocks:
                x = blk(x, rope=rot_pos_embed)
                x_no_cls = x[:, 1:, ...]  # Assuming the cls token is the first token as usual

                if len(x_no_cls.shape) == 3:
                    stride = max(1, x_no_cls.shape[1] // 4)
                    pooled_output = F.avg_pool1d(x_no_cls.permute(0, 2, 1), kernel_size=stride, stride=stride).permute(0, 2, 1)
                elif len(x_no_cls.shape) == 4:
                    stride = max(1, x_no_cls.shape[2] // 4)
                    pooled_output = F.avg_pool2d(x_no_cls, kernel_size=(stride, stride), stride=stride)
                else:
                    raise ValueError(f"Unexpected tensor shape: {x_no_cls.shape}")

                flattened_output = pooled_output.reshape(pooled_output.size(0), -1)
                feature_list.append(flattened_output)

            outcome = torch.cat(feature_list, 1)
            return outcome

    base_model = timm.create_model(
        'eva02_large_patch14_448.mim_m38m_ft_in22k',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    base_model = base_model.eval().cuda()

    # Wrap the base model in your custom class
    model = CustomModelWrapper(base_model)
    data_config = timm.data.resolve_model_data_config(base_model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model, transform

def load_custom_vit(args):
    base_model = timm.create_model(
        args.arch,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    base_model = base_model.eval().cuda()

    data_config = timm.data.resolve_model_data_config(base_model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return base_model, transform

def extract_features_from_vit(model, dl, args, extracted_train_path, labels_train_path, max_count=10**9):
    chunk_size = 1024
    current_count = 0

    existing_chunks = glob.glob(f"{extracted_train_path}_chunk_*.pt")
    chunk_idx = 0
    already_processed = 0

    if existing_chunks:
        last_saved_chunk = max(existing_chunks, key=os.path.getctime)
        chunk_idx = int(last_saved_chunk.split("_chunk_")[-1].split('.pt')[0]) + 1
        already_processed = chunk_idx * chunk_size
        print(f"Found {len(existing_chunks)} existing chunks. Starting from chunk {chunk_idx}")
        # Create a subset of the dataset starting from the already_processed index
        subset_dataset = Subset(dl.dataset, range(already_processed, len(dl.dataset)))
        subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=dl.batch_size, shuffle=False, num_workers=dl.num_workers)
        dl = subset_dataloader

    # Initialize empty tensors to store the current chunk
    feat_chunk = torch.Tensor().cpu()
    labels_chunk = torch.Tensor().cpu()

    for img, label in tqdm(dl):
        img = img.cuda()  # Move data to CUDA device
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model_out = model.forward_features(img).detach().cpu().float()  # Use wrapped model for DataParallel

        # Concatenate the new data to the existing chunk
        feat_chunk = torch.cat((feat_chunk, model_out), dim=0)
        labels_chunk = torch.cat((labels_chunk, label.cpu().float()), dim=0)

        current_count += len(img)

        # Check if the current chunk has reached the chunk_size
        if feat_chunk.shape[0] >= chunk_size:
            print(f"Saving chunk {chunk_idx}")
            torch.save(feat_chunk, f"{extracted_train_path}_chunk_{chunk_idx}.pt")
            torch.save(labels_chunk, f"{labels_train_path}_chunk_{chunk_idx}.pt")

            # Reset the chunks
            feat_chunk = torch.Tensor().cpu()
            labels_chunk = torch.Tensor().cpu()
            chunk_idx += 1  # Increment chunk index

        if current_count >= max_count:
            break

    # Save the last chunk
    if feat_chunk.shape[0] > 0:
        print(f"Saving chunk {chunk_idx}")
        torch.save(feat_chunk, f"{extracted_train_path}_chunk_{chunk_idx}.pt")
        torch.save(labels_chunk, f"{labels_train_path}_chunk_{chunk_idx}.pt")

def check_features_extracted(base_path):
    print(f"Checking if features exist at {base_path}")
    # Check for chunk files
    chunk_files_exist = len(glob.glob(f"{base_path}_chunk_*.pt")) > 0
    if chunk_files_exist:
        return True

    # Check for a single file
    single_file_exist = os.path.isfile(base_path)
    return single_file_exist

def load_chunk(chunk_file, dtype):
    try:
        chunk = torch.load(chunk_file).to(dtype=dtype)
        return chunk
    except Exception as e:
        print(f"Error loading {chunk_file}: {e}")
        return None
def load_chunks(path_pattern, dtype, start_idx, end_idx, total_chunks, seed=None, is_training=False):
    chunk_files = sorted(glob.glob(path_pattern), key=lambda x: int(x.split("_chunk_")[-1].split('.pt')[0]))

    if is_training and seed is not None:
        np.random.seed(seed)
        # Generate random indices for the total number of chunks
        all_indices = np.random.permutation(total_chunks)
        # Select the indices for the current subset
        selected_indices = all_indices[start_idx:end_idx]
        # Filter chunk_files based on the selected indices
        chunk_files = [chunk_files[i] for i in selected_indices]

    chunks = []
    for chunk_file in tqdm(chunk_files, desc="Loading Chunks"):
        try:
            chunk = torch.load(chunk_file).to(dtype=dtype)
            chunks.append(chunk)
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")

    return torch.cat(chunks, dim=0) if chunks else torch.Tensor()
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def load_chunks(feature_pattern, label_pattern, dtype, start_idx, end_idx, total_chunks, seed=None, is_training=False, max_workers=1):
#     feature_files = sorted(glob.glob(feature_pattern), key=lambda x: int(x.split("_chunk_")[-1].split('.pt')[0]))
#     label_files = sorted(glob.glob(label_pattern), key=lambda x: int(x.split("_chunk_")[-1].split('.pt')[0]))

#     if is_training and seed is not None:
#         np.random.seed(seed)
#         all_indices = np.random.permutation(total_chunks)
#         selected_indices = all_indices[start_idx:end_idx]
#         feature_files = [feature_files[i] for i in selected_indices]
#         label_files = [label_files[i] for i in selected_indices]

#     features = []
#     labels = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_feature_chunk = {executor.submit(load_chunk, chunk_file, dtype): chunk_file for chunk_file in feature_files}
#         future_to_label_chunk = {executor.submit(load_chunk, chunk_file, torch.long): chunk_file for chunk_file in label_files}
        
#         for future in as_completed(future_to_feature_chunk):
#             feature_chunk = future.result()
#             if feature_chunk is not None:
#                 features.append(feature_chunk)
                
#         for future in as_completed(future_to_label_chunk):
#             label_chunk = future.result()
#             if label_chunk is not None:
#                 labels.append(label_chunk)

#     if features and labels:
#         return torch.cat(features, dim=0), torch.cat(labels, dim=0)
#     else:
#         return torch.Tensor(), torch.Tensor()

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
    labels_train_path = extracted_path + "/_train_labels.npy"
    labels_test_path = extracted_path + "/_test_labels.npy"
    
    kwargs = {"num_workers": args.workers, "pin_memory": True}
    if not os.path.exists(dataset_path):
        raise Exception(
            "We cannot download a dataset/model here \n Run python utils.py to download things"
        )
    if args.dataset in ["ImageNet"]:
        def get_feature_path(base_path, feature_mod, default_suffix=".npy"):
            if feature_mod != "none":
                return base_path.replace(default_suffix, f"{feature_mod}.npy")
            return base_path

        # Determine paths for extracted features and labels
        extracted_train_path = get_feature_path(extracted_path + "/_train.npy", args.feature_mod)
        labels_train_path = extracted_path + "/_train_labels.npy"
        extracted_test_path = get_feature_path(extracted_path + "/_test.npy", args.feature_mod)
        labels_test_path = extracted_path + "/_test_labels.npy"

        if check_features_extracted(extracted_train_path) and not args.overwrite:
        # if False:
            print(f"Features already extracted for {extracted_train_path}, skipping...")
        else:
            print(f"Extracting features to {extracted_train_path}...")
            if "eva" in args.arch:
                model, transform = load_custom_eva(args)
            elif "vit" in args.arch:
                model, transform = load_custom_vit(args)
                # below code is for dprandp
                # model = load_pretrained_vit(args) 
                # IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
                # IMAGENET_STD = np.array([0.229, 0.224, 0.225])
                # transform = transforms.Compose([
                #         transforms.Resize(256, interpolation=3),
                #         transforms.CenterCrop(224),
                #         transforms.ToTensor(),
                #         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])     
            elif "resnet" in args.arch:
                model = create_model_from_pretrained(args)
            # model.eval()
            # we have 2 gpus so let's use DataParallel
            # model = nn.DataParallel(model)
            # model = model.cuda()
             
            dataset_path = args.dataset_path + args.dataset
            
            # train_path = dataset_path + "/train"
            # train_ds = datasets.ImageFolder(train_path, transform=transform_train)
            train_ds = datasets.ImageNet(root="/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization", split="train", transform=transform)
            batch_extract_size = 32
            train_loader = DataLoader(train_ds, batch_size=batch_extract_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            os.makedirs(extracted_path, exist_ok=True)
            # Actual extraction
            max_count_num = 10**9
            extract_features_from_vit(model, train_loader, args, extracted_train_path, labels_train_path, max_count=max_count_num)

            # test_path = dataset_path + "/val"
            # test_ds = datasets.ImageFolder(test_path, transform=transform_val)
            test_ds = datasets.ImageNet(root="/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization", split="val", transform=transform)
            test_loader = DataLoader(test_ds, batch_size=batch_extract_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            # features_test, labels_test = extract_features_from_vit(model, test_loader, args, max_count=10**9)
            extract_features_from_vit(model, test_loader, args, extracted_test_path, labels_test_path, max_count=max_count_num)
        print("Loading features from disk")

        # Define the range of chunks to load
        start_idx = args.start_idx  # Replace with your actual start index
        end_idx = args.end_idx  # Replace with your actual end index
        seed = args.seed
        features_dtype = torch.float16  # or torch.float32 based on your preference
        total_chunks = 1251
        # Load feature and label chunks
        # features_train, labels_train = load_chunks(extracted_train_path + "_chunk_*.pt",
        #                                         labels_train_path + "_chunk_*.pt",
        #                                         features_dtype, start_idx, end_idx,
        #                                         total_chunks, seed=seed, is_training=True, max_workers=args.workers)

        # features_test, labels_test = load_chunks(extracted_test_path + "_chunk_*.pt",
        #                                         labels_test_path + "_chunk_*.pt",
        #                                         features_dtype, 0, 49, 49, max_workers=args.workers)
        # # Load feature chunks
        
        features_train = load_chunks(extracted_train_path + "_chunk_*.pt", features_dtype, start_idx, end_idx, 1252, seed=args.seed, is_training=True)
        features_test = load_chunks(extracted_test_path + "_chunk_*.pt", features_dtype, 0, 49, 49)
        # # Load label chunks
        labels_dtype = torch.long
        labels_train = load_chunks(labels_train_path + "_chunk_*.pt", labels_dtype, start_idx, end_idx, 1252, seed=args.seed, is_training=True)
        labels_test = load_chunks(labels_test_path + "_chunk_*.pt", labels_dtype, 0, 49, 49)
        
        def preprocess_features_tensor(features_tensor, desired_norm, noise_std=0.0, chunk_size=10000):
            num_samples, num_features = features_tensor.shape
            num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Compute the number of chunks
            
            normalized_features_list = []
            mean_features_list = []
            
            # First pass: Compute mean and normalize the features
            for i in tqdm(range(num_chunks), desc="Processing Chunks"):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, num_samples)
                
                # Process each chunk
                chunk = features_tensor[start_idx:end_idx].float()  # Convert only this chunk to float32
                chunk_norms = torch.norm(chunk, dim=1, keepdim=True)  # Compute norms of each vector in the chunk
                
                # Normalize the chunk: each vector in the chunk is scaled to have the norm 'desired_norm'
                normalized_chunk = (chunk / chunk_norms * desired_norm)
                mean_features_list.append(normalized_chunk.mean(dim=0))  # Normalize and convert back to fp16 immediately
                normalized_features_list.append(normalized_chunk.half())
                  # Compute the mean of the chunk

            # # Compute the overall mean in fp16
            mean_feature_vector = torch.mean(torch.stack(mean_features_list), dim=0)

            # # Add noise to the mean vector in fp16
            # noise = torch.normal(mean=0, std=noise_std, size=mean_feature_vector.shape, device=mean_feature_vector.device).half()
            # noisy_mean_vector = mean_feature_vector + noise

            # # Second pass: Subtract the noisy mean from the normalized features
            for i, normalized_chunk in enumerate(normalized_features_list):
                normalized_chunk = normalized_chunk.float() - mean_feature_vector  # Subtract the noisy mean
                normalized_features_list[i] = normalized_chunk.half()

            # Concatenate all normalized chunks
            normalized_features = torch.cat(normalized_features_list, dim=0)
            
            return normalized_features

        train_noise_std = 0.0
        if args.feature_norm != 0:  # Adjusted condition to cater for -1 as a valid input
            features_train = preprocess_features_tensor(features_train, desired_norm=args.feature_norm, noise_std=train_noise_std) # this noise is already calibrated for eps guarantee
            features_test = preprocess_features_tensor(features_test, desired_norm=args.feature_norm, noise_std=0.0) # no noise for test set

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
                args, images_train=None, images_test=images_test, labels_train=None, labels_test=labels_test
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
                args, images_train=None, images_test=images_test, labels_train=None, labels_test=labels_test
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
            extract_features(args, images_train=None, images_test=images_test, labels_train=None, labels_test=labels_test)
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
        # THIS IS HARDCODED TO WORK FOR THE GITHUB FEATURES
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
            extract_features(args, images_train, images_test, labels_train, labels_test) # if you get any errors around here it may because I had to fiddle with things to upload the extracted features to GitHub
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