from __future__ import annotations

import logging
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from torch import nn
import numpy as np

def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generate
            (see the notes)

    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).

        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )



def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.

    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.

    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``

    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor

    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor or a list of tensors

    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)

def _randk(k:int, vec: torch.Tensor) -> torch.Tensor:
    """
    Performs the gradient sparsification algorithm described in Wangni et al 2018
    """
    ones = torch.ones_like(vec)
    density = torch.min(k * vec.abs()/vec.sum(), ones)
    for _ in range(4):
        residual_density = density[(density != 1).nonzero(as_tuple=True)]
        density = torch.where(density >= ones, ones, density)
        if len(residual_density) == 0 or residual_density.abs().sum() == 0:
            break
        else:
            c = (k - len(vec) + len(residual_density))/residual_density.sum()
        density = torch.min(c * density, ones)
        if c <= 1:
            break
    sample = torch.rand_like(vec)
    rescaled = torch.where(density <= 10**(-6), vec, vec / density)
    ret = torch.where(sample <= density, rescaled, torch.zeros_like(vec))
    ret = ret.reshape(vec.shape)
    return ret

def prep_grad(x):
    x_flat = torch.unsqueeze(x, 0).flatten()
    dim = x.shape
    d = x_flat.shape[0]
    return x_flat, dim, d

def grad_spars_opt(x, h, max_it):
    """
    :param x: vector to sparsify
    :param h: density
    :param max_it: maximum number of iterations of greedy algorithm
    :return: compressed vector
    """
    x, dim, d = prep_grad(x)
    # number of coordinates kept
    r = int(np.maximum(1, np.floor(d * h)))

    abs_x = torch.abs(x)
    abs_sum = torch.sum(abs_x)
    ones = torch.ones_like(x)
    p_0 = r * abs_x / abs_sum
    p = torch.min(p_0, ones)
    for _ in range(max_it):
        p_sub = p[(p != 1).nonzero(as_tuple=True)]
        p = torch.where(p >= ones, ones, p)
        if len(p_sub) == 0 or torch.sum(torch.abs(p_sub)) == 0:
            break
        else:
            c = (r - d + len(p_sub))/torch.sum(p_sub)
        p = torch.min(c * p, ones)
        if c <= 1:
            break
    prob = torch.rand_like(x)
    # avoid making very small gradient too big
    s = torch.where(p <= 10**(-6), x, x / p)
    # we keep just coordinates with high probability
    t = torch.where(prob <= p, s, torch.zeros_like(x))
    t = t.reshape(dim)
    return t

def _topk(sparsity:float, vec: torch.Tensor, do_induced: bool) -> torch.Tensor:
    """
    Returns a vector of the same size as params, but with all 
    coordinates that are not in the topk zeroed out
    Args:
        k: number of params to use
        vec: nn parameters
        do_induced: whether to do induced compression

    Returns:
        modified vec vector
    """
    k = int(sparsity * len(vec))
    topkVals = torch.zeros(k, device=vec.device)
    topkIndices = torch.zeros(k, device=vec.device).long()
    torch.topk(vec**2, k, sorted=False, out=(topkVals, topkIndices))

    ret = torch.zeros_like(vec)
    ret[topkIndices] = vec[topkIndices]
    # if do_induced:
    #     """
    #     Compute the residual vector
    #     Then, add the rand-k of the residual to ret
    #     """
    #     if k > len(vec)//2:
    #         return vec
    #     residual = torch.clone(vec)
    #     residual[ret!=0] = 0
    #     rand_k = _randk(k, residual)
    #     ret += rand_k
    return ret

def _uncompressed_mask(x: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[ind] = 1
    t = mask * x
    return t

def generate_topk_ref(k:float, params: List[nn.Parameter], do_induced: bool, do_baseline: bool=True) -> torch.Tensor:
    """
    """
    # return _topk(k, get_grad_vec(params), do_induced)
    x = get_grad_vec(params, do_baseline)
    return generate_topk(k, x, do_induced)

def generate_topk_ref_with_topk(k:float, params: List[nn.Parameter], curr_k: torch.Tensor, do_induced: bool, do_baseline: bool=True) -> torch.Tensor:
    """
    """
    x = get_grad_vec(params, do_baseline)
    c_1 = _uncompressed_mask(x, curr_k)
    return generate_topk_with_topk(k, x, c_1, do_induced)

def top_k_opt(x, h):
    """
    :param x: vector to sparsify
    :param h: density
    :return: compressed vector
    """
    x, dim, d = prep_grad(x)
    # number of coordinates kept
    r = int(np.maximum(1, np.floor(d * h)))
    # positions of top_k coordinates
    vals, ind = torch.topk(torch.abs(x), r)
    # print("MAX", vals[0])
    # print("GAP", np.abs((vals[-2] - vals[-1]).cpu()))
    t = _uncompressed_mask(x, ind)
    t = t.reshape(dim)
    return t

def generate_topk_with_topk(k:float, x: torch.Tensor, c_1: torch.Tensor, do_induced:bool) -> torch.Tensor:
    if do_induced:
        error = x - c_1
        c_2 = grad_spars_opt(error, k, 4)
        return c_1, c_1 + c_2
    else:
        return c_1, c_1
    
def generate_topk(k:float, x: torch.Tensor, do_induced:bool) -> torch.Tensor:
    # c_1 = _topk(k, x, False)
    c_1 = top_k_opt(x, k)
    return generate_topk_with_topk(k, x, c_1, do_induced)

def get_param_vec(params: List[nn.Parameter]) -> torch.Tensor:
    param_vec = []
    for p in params:
        param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)

def get_grad_vec(params: List[nn.Parameter], do_baseline:bool=True) -> torch.Tensor:
    grad_vec = []
    for p in params:
        if do_baseline:
            grad_vec.append(p.summed_grad.view(-1).float())
        else:
            grad_vec.append(p.grad.view(-1).float())
    return torch.cat(grad_vec)

def generate_noise_topk(
    std: float,
    reference: torch.Tensor,
    topk_slice: torch.Tensor,
    start: int,
    end: int,
    secure_mode: bool,
    generator=None,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        k: number of parameters to use
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    noise = torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
        generator=generator,
    )
    noise[topk_slice==0] = 0
    return noise