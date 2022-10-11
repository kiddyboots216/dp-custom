from typing import List, Optional, Tuple, Union

from opacus.accountants import IAccountant
from opacus.accountants.analysis import  rdp as privacy_analysis
import torch
import numpy as np
from opacus.optimizers import DPOptimizer
from typing import Callable
from tqdm import tqdm
import pickle

from os.path import exists
import pdb
import code

class MyPdb(pdb.Pdb):
    def do_interact(self, arg):
        code.interact("*interactive*", local=self.curframe_locals)

class InfluenceBoundedRDPAccountant(IAccountant):
    """
    This class implements the subsampled 
    """
    DEFAULT_ALPHAS = [1.0+x/100.0 for x in range(1,100)]+[2.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self, 
                 epsilon, 
                 delta, 
                 model, 
                 sample_rate, 
                 optimizer, 
                 n_data,
                 alphas: Optional[List[Union[float, int]]] = None,
                 l2_norm_budget: bool = False,
        ):
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
        self.n_data = n_data
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.delta = delta

        if alphas==None:
            self.alphas=self.DEFAULT_ALPHAS
        else:
            self.alphas=alphas
        if l2_norm_budget and sample_rate == 1:
            self.norm_budget = self.get_l2_norm_budget()
            self.privacy_spent=torch.zeros(self.n_data, device=self.device)
            self.privacy_accumulate = lambda x, y: self.gaussian_accumulate(x, y)
            self.privacy_step = lambda x: x
        else:
            self.rdp_budget=get_rdp_budget(epsilon=self.epsilon,delta=self.delta,orders=alphas).to(self.device)
            self.privacy_spent=torch.zeros(size=(self.n_data, len(self.alphas)), device=self.device)
            self.privacy_usage=torch.zeros(self.n_data, device=self.device)
            self.privacy_accumulate = lambda x, y: x + y
            self.privacy_step = self.get_rdp_spent
        self.max_grad_norm = optimizer.max_grad_norm
        self.cache = torch.ones(self.n_data, device=self.device) * self.max_grad_norm
        self.sample_rate=sample_rate
        self.violations=torch.zeros(self.n_data, device=self.device)
        self.rdp_dict=self.pre_compute_rdp()

    def get_l2_norm_budget(self):
        # hardcoded for now to just not really do budgeting and just record
        budget = 360000
        return torch.ones(self.n_data, device=self.device) * budget

    def pre_compute_rdp(self,):
        file_name="rdp_calculation_"
        file_name+="%.3f" %self.sample_rate
        file_name+=".dic"

        if exists(file_name):
            with open(file_name, 'rb') as handle:
                rdp_dict=pickle.load(handle)
        else:
            noise_multiplier=self.optimizer.noise_multiplier
            rdp_dict=dict()
            print("pre-calculating rdp")
            NUMBER_OF_MULTIPLIERS=10000
            for i in range(NUMBER_OF_MULTIPLIERS):
                # if i%100==0:
                #     print(i)
                #     # print(rdp_dict)
                key_i= "%.2f"%(0.01+i/100)
                rdp_dict[key_i]=privacy_analysis.compute_rdp(
                                q=self.sample_rate,
                                noise_multiplier=0.01+i/100,
                                steps=1,
                                orders=self.alphas,
                            )
            # print(rdp_dict)
            rdp_dict['max']=rdp_dict["%.2f"%(0.01+(NUMBER_OF_MULTIPLIERS-1)/100)]
            # print(rdp_dict)
            with open(file_name, 'wb') as handle:
                pickle.dump(rdp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return rdp_dict
        
    def get_privacy_violations(self, privacy_spent):
        """
        1) Compute whether the privacy spent is greater than the budget
        2) Return the violations
        """
        violations = torch.prod(privacy_spent>self.rdp_budget, axis=1)
        return violations
        
    def get_privacy_usage(self, privacy_spent):
        """
        1) Compute the privacy usage at each RDP order
        2) Return the minimum usage across orders
        """
        privacy_usage = torch.min(privacy_spent/self.rdp_budget, dim=1)[0]
        return privacy_usage

    def cache_norms(self, batch_indices, norms):
        self.cache[batch_indices] = norms
        
    def retrieve_norms(self):
        return self.cache

    def compute_norms(self, batch_indices):
        """
        Computes the norms from what is stored in optimizer
        Stores the output in cache
        """
        per_sample_norms = self.optimizer.per_sample_norms
        per_sample_norms = torch.minimum(per_sample_norms, torch.ones_like(per_sample_norms) * self.max_grad_norm)
        self.cache_norms(batch_indices, per_sample_norms)

    def step(self):
        """
        1) Spend privacy based on the cached norms; "if I see this gradient I will clip it to this value"
        2) Compose this privacy with your privacy spent so far
        3) Update violations based on what budget you have exceeded; this will overspend
        """
        per_sample_norms = self.retrieve_norms()
        per_sample_noise_multiplier = self.optimizer.noise_multiplier * self.max_grad_norm/per_sample_norms
        privacy_current_step = self.privacy_step(per_sample_noise_multiplier)
        # privacy_current_step = self.get_rdp_spent(per_sample_noise_multiplier=per_sample_noise_multiplier)
        self.privacy_spent = self.privacy_accumulate(self.privacy_spent, privacy_current_step)
        # self.privacy_spent += privacy_current_step # can overspend but that's ok
        self.violations = self.get_privacy_violations(self.privacy_spent) # update violations
        self.privacy_usage = self.get_privacy_usage(self.privacy_spent) # minimize over RDP orders, this is just for reporting
        
    def get_violations(self, batch_indices):
        """
        Return the indices of violations in the current batch
        """
        violations = self.violations[batch_indices]
        return violations

    def print_stats(self):
        print(f"Violations {torch.sum(self.violations)} \t Max/Mean/Min Privacy {torch.max(self.privacy_usage)} | {torch.mean(self.privacy_usage)} | {torch.median(self.privacy_usage)}")
        
    def get_rdp_spent(
        self, per_sample_noise_multiplier,
    ):
        per_sample_noise_multiplier=per_sample_noise_multiplier.cpu().numpy()
        privacy_spent = torch.zeros(size = (self.n_data, len(self.alphas)), device=self.device)
        for i in range(self.n_data):
            noise = per_sample_noise_multiplier[i]
            if(self.violations[i] or noise == 0):
                privacy_spent[i]=0
            else:
                privacy_spent[i] = self.get_rdp(noise)
        return privacy_spent


    def get_rdp(self, noise_multiplier):
        assert noise_multiplier != 0
        if self.rdp_dict:
            if "%.2f"%noise_multiplier in self.rdp_dict.keys():
                return torch.tensor(self.rdp_dict["%.2f"%noise_multiplier])
            else:
                return torch.tensor(self.rdp_dict["max"])
        else:
            return torch.tensor(
                privacy_analysis.compute_rdp(
                q=self.sample_rate,
                noise_multiplier=noise_multiplier,
                steps=1,
                orders=self.alphas,
            )
            )

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        return self.epsilon

    def __len__(self):
        return None

    @classmethod
    def mechanism(cls) -> str:
        return "rdp_influence_bounded"



def get_rdp_budget(
    *, orders: Union[List[float], float], epsilon: float, delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.zeros(len(orders_vec))
    eps_vec= np.ones(len(orders_vec))*epsilon
    # print(orders_vec,rdp_vec,eps_vec)
    # print((np.log(delta)))
    # print((np.log(delta)) / (orders_vec - 1),  (np.log(orders_vec)) / (orders_vec - 1)
    #     - np.log((orders_vec - 1) / orders_vec))
    rdp_budget = (
        eps_vec
        + (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        - np.log((orders_vec - 1) / orders_vec)
    )
    # print(rdp_budget)

    return torch.tensor(rdp_budget).float().clamp(min=1e-5)


