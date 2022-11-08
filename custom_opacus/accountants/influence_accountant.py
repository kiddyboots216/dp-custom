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

class FilterAccountant(IAccountant):
    def __init__(self, 
                 epsilon, 
                 delta, 
                 sample_rate, 
                 optimizer,
                 l2_norm_budget: bool = False,
        ):
        self.epsilon = epsilon
        self.delta = delta
        self.sample_rate = sample_rate
        self.optimizer = optimizer
        self.max_grad_norm = self.optimizer.max_grad_norm
        if l2_norm_budget:
            self.l2_squared_budget = self.get_l2_norm_budget()
            self.max_grad_norm = np.minimum(self.max_grad_norm, np.sqrt(self.l2_squared_budget))
            self.privacy_spent = 0
            self.privacy_step = lambda x: x**2
            self.update_max_grad_norm = self._update_max_grad_norm
            self.update_violations = lambda *args, **kwargs: None
            self.get_privacy_usage = lambda: self.privacy_spent / self.l2_squared_budget

    def get_l2_norm_budget(self):
        return cal_overall_norm_budget(self.optimizer.noise_multiplier * self.optimizer.max_grad_norm, self.epsilon, self.delta)**2
    
    def _update_max_grad_norm(self):
        self.max_grad_norm = np.minimum(self.max_grad_norm, np.sqrt(self.l2_squared_budget - self.privacy_spent))

    def step(self):  
        self.privacy_spent += self.privacy_step(self.max_grad_norm)
        self.update_max_grad_norm()
        
    @property
    def privacy_usage(self):
        return self.get_privacy_usage()
    
    def get_epsilon(
        self, *args, **kwargs,
    ):
        return self.epsilon
    
    @classmethod
    def mechanism(cls) -> str:
        return "gdp_filter"
    
    def __len__(self):
        return None
    
class InfluenceBoundedRDPAccountant(IAccountant):
    DEFAULT_ALPHAS = [1.0+x/100.0 for x in range(1,100)]+[2.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self, 
                 epsilon, 
                 delta, 
                 sample_rate, 
                 optimizer, 
                 n_data,
                 alphas: Optional[List[Union[float, int]]] = None,
                 l2_norm_budget: bool = False,
        ):
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
        self.n_data = n_data
        self.device = "cuda:0"
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.delta = delta
        self.base_grad_norm = self.optimizer.max_grad_norm
        self.max_grad_norm = torch.ones(self.n_data, device=self.device) * self.base_grad_norm
        self.cache = torch.ones(self.n_data, device=self.device) * self.base_grad_norm
        self.sample_rate = sample_rate
        self.violations = torch.zeros(self.n_data, device=self.device)
        if alphas==None:
            self.alphas=self.DEFAULT_ALPHAS
        else:
            self.alphas=alphas
        if l2_norm_budget and sample_rate == 1: # we will use GDP
            self.l2_squared_budget = self.get_l2_norm_budget()
            self.max_grad_norm = torch.minimum(self.max_grad_norm, torch.sqrt(self.l2_squared_budget))
            self.privacy_spent = torch.zeros(self.n_data, device=self.device)
            self.privacy_step = lambda x: x**2
            self.update_max_grad_norm = self._update_max_grad_norm
            self.update_violations = lambda *args, **kwargs: None
            self.get_privacy_usage = lambda: self.privacy_spent.max() / self.l2_squared_budget
        else: # we will use RDP
            self.rdp_budget=get_rdp_budget(epsilon=self.epsilon,delta=self.delta,orders=alphas).to(self.device)
            self.privacy_spent = torch.zeros(size=(self.n_data, len(self.alphas)), device=self.device)
            self.privacy_step = lambda x: self.get_rdp_spent(self.optimizer.noise_multiplier * self.max_grad_norm/x)
            self.update_max_grad_norm = lambda *args, **kwargs: None
            self.update_violations = self.get_privacy_violations
            self.get_privacy_usage = lambda: torch.min(self.privacy_spent/self.rdp_budget, dim=1)[0]
            self.rdp_dict = self.pre_compute_rdp()


    def get_l2_norm_budget(self):
        return torch.ones(self.n_data, device=self.device) * cal_overall_norm_budget(self.optimizer.noise_multiplier * self.optimizer.max_grad_norm, self.epsilon, self.delta)**2
        
    def get_privacy_violations(self):
        self.violations = torch.prod(self.privacy_spent>self.rdp_budget, axis=1)

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
        per_sample_norms = torch.minimum(per_sample_norms, self.max_grad_norm[batch_indices])
        self.cache_norms(batch_indices, per_sample_norms)

    def remaining_norm(self):
        return self.max_grad_norm[self.max_grad_norm>0].shape.numel()

    def step(self):
        """
        1) Determine per sample clipping thresholds (done by optimizer)
        2) Determine what we can afford
        3) Clip according to what we can afford and update privacy spent accordingly
        """            
        self.privacy_spent += self.privacy_step(self.retrieve_norms())
        self.update_max_grad_norm() # does nothing for RDP, only does something for GDP
        self.update_violations() # does nothing for GDP, only does something for RDP
        # self.violations = self.get_privacy_violations(self.privacy_spent) # update violations
        # self.privacy_usage = self.get_privacy_usage(self.privacy_spent) # minimize over RDP orders, this is just for reporting
    
    def _update_max_grad_norm(self):
        # assert self.l2_squared_budget > self.privacy_spent # measure twice, cut once
        self.max_grad_norm = torch.minimum(self.max_grad_norm, torch.sqrt(self.l2_squared_budget - self.privacy_spent))

    @property
    def privacy_usage(self):
        return self.get_privacy_usage()
    
    def get_violations(self, batch_indices):
        """
        Return the indices of violations in the current batch
        """
        violations = self.violations[batch_indices]
        return violations

    def print_stats(self):
        privacy_usage = self.privacy_usage
        print(f"Violations {torch.sum(self.violations)} \t Max/Mean/Med Privacy {torch.max(privacy_usage)} | {torch.mean(privacy_usage)} | {torch.median(privacy_usage)}")
        
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


from opacus.accountants.analysis import rdp
from scipy.stats import norm

DEFAULT_ALPHAS = [1 + x / 100.0 for x in range(1, 300)] + list(range(12, 64))
DEFAULT_delta=1e-5
from scipy.stats import norm, binom
import torch

def cal_overall_norm_budget(sigma, epsilon,delta):
    return sigma/search_noise_mul_for_delta(epsilon, delta)

def cal_delta_gaussian_full(noise_mul, epsilon):
    u=1/noise_mul
    return norm.cdf(-epsilon/u +u/2) - np.exp(epsilon)*norm.cdf(-epsilon/u -u/2)



def search_noise_mul_for_delta(epsilon, delta):
    noise_mul=1.0
    if cal_delta_gaussian_full(noise_mul, epsilon)> delta:
        while cal_delta_gaussian_full(noise_mul, epsilon)> delta:
            noise_mul=noise_mul*2

        noise_mul_lower=noise_mul/2
        noise_mul_upper=noise_mul
    else:
        while cal_delta_gaussian_full(noise_mul, epsilon)> delta:
            noise_mul=noise_mul/2

        noise_mul_lower=noise_mul
        noise_mul_upper=noise_mul*2


    while noise_mul_upper-noise_mul_lower> 0.00001:
        noise_mul_middle=(noise_mul_upper+noise_mul_lower)/2
        delta_middle= cal_delta_gaussian_full(noise_mul_middle,epsilon)
        if delta_middle> delta:
            noise_mul_lower=noise_mul_middle
        else:
            noise_mul_upper=noise_mul_middle

    return noise_mul_upper


def cal_epsilon_delta_with_rdp_list(n, noise_mul, q):
    rdp_vec =rdp.compute_rdp(
                q=q[0],
                noise_multiplier=noise_mul[0],
                steps=n[0],
                orders=DEFAULT_ALPHAS,)
    for i in range(1, len(n)):
        rdp_vec=rdp_vec+ rdp.compute_rdp(
                q=q[i],
                noise_multiplier=noise_mul[i],
                steps=n[i],
                orders=DEFAULT_ALPHAS,)
    # print(rdp_vec)
    eps, best_alpha = rdp.get_privacy_spent(
        orders=DEFAULT_ALPHAS, rdp=rdp_vec, delta=DEFAULT_delta)
    # print(best_alpha)
    return(eps,DEFAULT_delta)


def cal_epsilon_gaussian_full(noise_mul, delta):

    epsilon=1.0
    if cal_delta_gaussian_full(noise_mul, epsilon)> delta:
        while cal_delta_gaussian_full(noise_mul, epsilon)> delta:
            epsilon=epsilon*2

        epsilon_lower=epsilon/2
        epsilon_upper=epsilon
    else:
        while cal_delta_gaussian_full(noise_mul, epsilon)<= delta:
            epsilon=epsilon/2

        epsilon_lower=epsilon
        epsilon_upper=epsilon*2

    while epsilon_upper-epsilon_lower> 0.00001:
        epsilon_middle=(epsilon_upper+epsilon_lower)/2
        delta_middle= cal_delta_gaussian_full(noise_mul,epsilon_middle)

        if delta_middle> delta:
            epsilon_lower=epsilon_middle
        else:
            epsilon_upper=epsilon_middle

    return epsilon_upper