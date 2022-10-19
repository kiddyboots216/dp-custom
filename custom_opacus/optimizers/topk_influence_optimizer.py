from __future__ import annotations

import torch
from torch.optim import Optimizer

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _generate_noise
from opacus.optimizers.optimizer import _check_processed_flag, _get_flat_grad_sample, _mark_as_processed

from opacus.grad_sample.grad_sample_module import GradSampleModule

import logging
from typing import Callable, List, Optional, Union

from torch import nn
import numpy as np
import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)
class SparsefluenceOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` compatible with
    distributed data processing
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        augmult: int = 0,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        self.augmult = augmult
        self.base_grad_norm = max_grad_norm
        print(f"AUGMULT {self.augmult}")

    def get_per_sample_norms(self):
        per_sample_grads=self.get_per_sample_grads()
        per_sample_norms = per_sample_grads.norm(2,dim=1)
        return per_sample_norms

    def get_per_sample_grads(self):
        per_sample_grads = torch.cat([g.view(len(g), -1) for g in self.grad_samples], dim=1)
        return per_sample_grads

    def compute_norms(self, violations):
        if self.augmult != 0:
            for p in self.params:
                p.grad_sample = p.grad_sample.view(p.grad_sample.shape[0]//16, 16, *(i for i in p.grad_sample.shape[1:])).sum(dim=1)
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        self.per_sample_norms = (1 - violations) * per_sample_norms
        
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        # n_layers, other stuff
        # other stuff = batch size, others
        per_sample_norms = self.per_sample_norms
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )
        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)

            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        1) Generate noise proportional to std
        2) Divide gradient by max grad norm
        3) Add noise to gradient
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            noise = _generate_noise(
                # std=self.noise_multiplier * self.max_grad_norm,
                std = self.noise_multiplier,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.summed_grad = p.summed_grad / self.base_grad_norm
            p.grad = (p.summed_grad + noise).view_as(p.grad)

            _mark_as_processed(p.summed_grad)