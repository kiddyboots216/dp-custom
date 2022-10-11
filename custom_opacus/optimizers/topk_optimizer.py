from __future__ import annotations

import logging
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer
import numpy as np

from opacus.optimizers import DPOptimizer
from opacus.optimizers.topk_utils import generate_topk_ref_with_topk, top_k_opt, get_grad_vec, generate_topk_ref, _generate_noise, _mark_as_processed, _check_processed_flag_tensor, _check_processed_flag, generate_noise_topk
# STUFF FOR TOPK

logger = logging.getLogger(__name__)
import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)

class DPOptimizerTopk(DPOptimizer):
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
        k: int = 0,
        do_induced: bool = False,
        do_baseline: bool = False,
        n_stale: int = 0,
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
        self.k = k
        self.d = None
        self.n_k = None
        self.do_baseline = do_baseline
        self.do_induced = do_induced
        self.n_stale = n_stale
        self.stale_counter = 0
        self.last_k = None
        # print(f"STALE COUNTER {self.stale_counter}")
        
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
            p.summed_grad = p.summed_grad / self.max_grad_norm
            p.grad = (p.summed_grad + noise).view_as(p.grad)

            _mark_as_processed(p.summed_grad)
        # """
        # Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        # 1) generate reference topk vector that is d-dimensional
        # 2) iterate through parameters; find the corresponding start and end
        # 3) get the topk indices and map them to corresponding elements of noise
        # 4) zero out noise vector in those indices, then add to .grad
        # """
        # if self.k == 1:
        #     super().add_noise()
        #     return
        # if self.k != 1:
        #     start = 0
        #     x = get_grad_vec(self.params)
        #     # have the potential to use the last_k for this update
        #     # MyPdb().set_trace()
        #     if self.stale_counter == 0:
        #         # print("RESETTING STALENESS")
        #         # reset stale counter and generate a new mask
        #         self.stale_counter = self.n_stale # this can be 0
        #         true_topk, topk_ref = generate_topk_ref(self.k, 
        #                                                 self.params, 
        #                                                 do_induced=self.do_induced)
        #         self.last_k = torch.nonzero(true_topk).squeeze()
        #         # self.stale_grad_buffer = torch.zeros_like(topk_ref)
        #     else:
        #         # decrement stale counter and use the stale mask
        #         self.stale_counter -= 1
        #         # print(f"USING {self.stale_counter}-STALE MASK")
        #         true_topk, topk_ref = generate_topk_ref_with_topk(self.k,
        #                                                         self.params,
        #                                                         self.last_k,
        #                                                         do_induced=self.do_induced)
        #     # MyPdb().set_trace()
        #     # self.stale_grad_buffer += 1/self.n_stale * x[self.last_k]
        #     if self.d is None:
        #         self.d = len(topk_ref)
        #         self.n_k = torch.count_nonzero(true_topk).item()
        #         # MyPdb().set_trace()
        #         print("**************SIZE OF MODEL IS****************", self.d)
        #         print("**************% PARAMS USED IS****************", self.n_k/self.d)
        #         # self.last_k = torch.zeros(self.n_k).to(topk_ref.device)
       
        # # x = get_grad_vec(self.params).abs().sort(descending=True)[0]
        # # ms = [100,200,400,800,1600]
        # # for m in ms:
        # #     print(f"SUM at {m} is {x[self.n_k - m : self.n_k + m].norm(p=2):.4f}")
        # #     print(f"COST at {m} is {(x[self.n_k - m : self.n_k] - x[self.n_k : self.n_k + m]).norm(p=2):.4f}")
        # #     print(f"GAP at {m} is {x[self.n_k - m] - x[self.n_k]:.4f} and {x[self.n_k] - x[self.n_k + m]:.4f}")
        # # print(f"RATIO {(x[self.n_k - m : self.n_k] - x[self.n_k : self.n_k + m]).norm(p=2) / x[self.n_k - m : self.n_k + m].norm(p=2)}")
        # # if self.last_k is not None:
        # #     # print(f"SIMILARITY {torch.sum(torch.isin(self.curr_k, self.last_k)).item() / len(self.curr_k):.2f}")
        # #     stale_k_idx = self.last_k
        # #     stale_k_vals = x[stale_k_idx]
        # #     stale_k = topk_ref.sort(descending=True)[0][:self.n_k]
        # #     true_k_idx = torch.nonzero(top_k_opt(x, self.k))
        # #     true_k_vals = x[true_k_idx]
        # #     assert len(stale_k_idx) == len(true_k_idx)
        # #     print(f"MASS PRESERVED {stale_k_vals.norm()/true_k_vals.norm():.2f}")
        #     # print(f"MASS PRESERVED {stale_k.norm()/true_k_vals.norm():.2f}")
        # # self.last_k = self.curr_k
        # # self.compute_cdist()
        # if self.do_baseline:
        #     for p in self.params:
        #         # end = start + p.numel()
        #         _check_processed_flag(p.summed_grad)
        #         # topk_slice = topk_ref[start:end].reshape(p.summed_grad.shape)
        #         noise = generate_noise_topk(
        #             # std=self.noise_multiplier * self.max_grad_norm,
        #             # MODIFIED TO FIT DEEPMIND ALGO
        #             std=self.noise_multiplier,
        #             reference=p.summed_grad,
        #             generator=self.generator,
        #             secure_mode=self.secure_mode,
        #             topk_slice=topk_slice,
        #             start=start,
        #             end=end,
        #         )
        #         # print(start, end)
        #         # start = end
        #         # print(torch.count_nonzero(noise))
        #         # p.summed_grad[topk_slice == 0] = 0
        #         p.summed_grad = p.summed_grad * 1/self.max_grad_norm # MODIFIED FOR DEEPMIND ALGO
        #         p.grad = (p.summed_grad + noise).view_as(p.grad)
        #         _mark_as_processed(p.summed_grad)
        # else:
        #     # first do the regular noise addition
        #     for p in self.params:
        #         _check_processed_flag(p.summed_grad)
        #         noise = _generate_noise(
        #             std=self.noise_multiplier * self.max_grad_norm,
        #             reference=p.summed_grad,
        #             generator=self.generator,
        #             secure_mode=self.secure_mode,
        #         )
        #         p.grad = (p.summed_grad + noise).view_as(p.grad)

        #         _mark_as_processed(p.summed_grad)
        #     # now what we have is private by postprocessing
        #     # THIS NEEDS TO ACCESS P.GRAD AND NOT P.SUMMED_GRAD
        #     true_topk, topk_ref = generate_topk_ref(self.k, self.params, do_induced=self.do_induced, do_baseline=False)
        #     start = 0
        #     for p in self.params:
        #         end = start + p.numel()
        #         topk_slice = topk_ref[start:end].view_as(p.grad)
        #         p.grad = topk_slice
        #         start = end
                
    # def zero_grad(self, **kwargs):
    #     super().zero_grad(**kwargs)
    #     self.curr_k = None
        
    def dump_grads(self, root_path):
        gs = torch.cat([g.view(len(g), -1) for g in self.grad_samples], dim=1)
        with open(root_path, 'wb') as f:
            for g in gs:
                np.save(f, g)
                
    def prep_gs(self):
        # MyPdb().set_trace()
        gs = torch.cat([g.view(len(g), -1) for g in self.grad_samples], dim=1)
        d = gs.shape[1]
        h = self.k
        r = int(np.maximum(1, np.floor(d * h)))
        normd_gs = gs * 1/gs.norm(dim=1)
        return normd_gs, d, r
                
    def compute_cdist(self):
        # this is broken
        normd_gs, d, r = self.prep_gs()
        leaveouts = []
        for idx, g in enumerate(normd_gs):                                                                         
            intermediate_leaveouts = []                                                                               
            for i, j in enumerate(normd_gs):                                                                          
                if idx != i:                                                                                             
                    intermediate_leaveouts.append(j)                                                                       
            leaveouts.append(torch.stack(intermediate_leaveouts).sum(axis=0))
        # sorteds = [torch.sort(agg.abs(), descending=True)[0] for agg in leaveouts]
        # topks = [g[:int(0.01 * len(g))] for g in sorteds]
        topks = [top_k_opt(g, h) for g in leaveouts]
        stacked_topks = torch.stack(topks)
        l2_diffs = torch.cdist(stacked_topks, stacked_topks)
        print(l2_diffs.flatten().sort(descending=True)[0][:20])
        
    def compute_budget_savings(self):
        # this is also broken
        normd_gs, d, r = self.prep_gs()
        true_topk_vals, ind = torch.topk(torch.stack(normd_gs).sum(dim=0).abs(), r)
        mask = torch.zeros(d)
        mask[ind] = 1
        budget_saved = torch.tensor([1 - (mask * g).norm() for g in normd_gs]).float()
        print(budget_saved.max(), budget_saved.min(), budget_saved.mean())
        
        