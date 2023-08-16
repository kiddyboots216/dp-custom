from prv_accountant import GaussianMechanism, PRVAccountant
from prv_accountant.dpsgd import find_noise_multiplier
import numpy as np
from functools import partial

"""
This file gives an example of the GDP composition that's used in the linear scaling rule
ts is an array of the number of iterations for different runs
We compute the corresponding noise multiplier that would be needed to reach a given target epsilon
This is a bit messy/slow, but we're using the find_noise_multiplier utility because it wraps Brent's method in a nice way and we don't have to worry about the details
We then compute the corresponding mus using Corollary 2.1 from our paper (which is itself just restating the GDP definition)
We then compute the total mu analytically using Corollary 3.3 from "Gaussian Differential Privacy"
We finally convert the mu into an epsilon using the PRV accountant
The final printed output is (lower bound on epsilon, estimate of epsilon, upper bound on epsilon) where the bounds are within eps_error of the estimate
"""

ts = [10, 75, 50, 25, 50, 45, 100] # change this however you want
# below are hardcoded values, you might also try eps_1 = 0.01, eps_2 = 0.1, eps_f = 0.97, etc.
eps_1 = 0.05
eps_2 = 0.10
eps_f = 0.96
eps_values = [eps_1] * 3 + [eps_2] * 3 + [eps_f]

find_noise_with_common_params = partial(
    find_noise_multiplier,
    sampling_probability=1,
    target_delta=1e-5,
    eps_error=0.001,
    mu_max=5000
)

searched_sigmas = [find_noise_with_common_params(num_steps=int(t), target_epsilon=eps) for t, eps in zip(ts, eps_values)]

print("Corresponding noise multipliers", searched_sigmas)
# compute mus for hyperparameter search
mus = []
for t, sigma in zip(ts, searched_sigmas):
    mu = np.sqrt(t/sigma**2)
    mus.append(mu)

# print(mus)
# compute total mu
mu_f = np.sqrt(sum([mu**2 for mu in mus]))

mech = GaussianMechanism(1) 
# override mu 
mech.mu = mu_f
# print(mech.mu)

eps_low, eps_est, eps_up = PRVAccountant(mech, max_self_compositions=1, eps_error=0.001,
                                  delta_error=1e-12).compute_epsilon(1e-5, 1)

print("Estimates of epsilon", eps_low, eps_est, eps_up)