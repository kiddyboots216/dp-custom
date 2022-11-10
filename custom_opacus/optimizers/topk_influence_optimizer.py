from __future__ import annotations

import torch
from torch.optim import Optimizer

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _generate_noise, _check_processed_flag, _mark_as_processed

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

    def compute_norms(self):
        if self.augmult > -1:
            for p in self.params:
                p.grad_sample = p.grad_sample.view(p.grad_sample.shape[0]//max(1,self.augmult), max(1,self.augmult), *(i for i in p.grad_sample.shape[1:])).mean(dim=1)
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        self.per_sample_norms = per_sample_norms
        
    def clip_and_accumulate(self, indices, violations):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        violations: 1 if ok to use this grad, else 0
        """
        per_sample_norms = self.per_sample_norms
        per_sample_clip_factor = (violations * self.max_grad_norm[indices] / (per_sample_norms + 1e-6)).clamp(
                max=1.0
        ) # if your max grad norm is 0, you are violated or not sampled
        # per_sample_clip_factor = per_sample_clip_factor / self.base_grad_norm
        for idx, p in enumerate(self.params):
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)
            # MyPdb().set_trace()

            # grad_sample = torch.einsum("i, i...->i...", 1/(self.max_grad_norm[indices] + 1e-6), grad_sample)
            # grad_sample = torch.einsum("i, i...->i...", 1/(per_sample_norms + 1e-6), grad_sample) BAD BAD

            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            # grad = grad / per_sample_norms
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
            print("GRAD NORM", p.summed_grad.norm())
            if p.summed_grad.norm() == 0:
                print("NO MORE OPTIMIZATION!")
                return True
            p.grad = (p.summed_grad + noise).view_as(p)
            print("NOISY GRAD NORM", p.grad.norm())

            _mark_as_processed(p.summed_grad)

    def pre_step(
        self, *args, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate(*args)
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, *args, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(*args):
            self.original_optimizer.step()
            return True
        else:
            return False

class DistributedSparsefluenceOptimizer(SparsefluenceOptimizer):
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
            augmult=augmult,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        # Noise only gets added to the first worker
        if self.rank == 0:
            super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

    def reduce_gradients(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(
        self, indices, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        if self.pre_step(indices):
            self.reduce_gradients()
            self.original_optimizer.step(closure)
            return True
        else:
            return False