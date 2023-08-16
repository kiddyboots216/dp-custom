from utils import DATASET_TO_SIZE, parse_args, get_ds
from prv_accountant import GaussianMechanism, PRVAccountant
from prv_accountant.dpsgd import find_noise_multiplier

import numpy as np
import copy

from submit_utils import launch_run

args = parse_args()

# CONSTANTS FOR HYPERPARAMETER SEARCH
# epsilon values - if you're asking why these sum up to greater than one it's because we are not using basic composition (we are using Corollary 3.3 from "Gaussian Differential Privacy" by Dong et al.)
# you will probably want to adjust these for your task, these are just some reasonable values for the current experiment
eps_1 = 0.01
eps_2 = 0.1
eps_f = 0.97 
# here are some other acceptable values that you can try out to see what happens when we give more privacy budget to the initial runs
# eps_1 = 0.05
# eps_2 = 0.1
# eps_f = 0.96 
# and some more
# eps_1 = 0.05
# eps_2 = 0.2
# eps_f = 0.9
eps_err = 0.001 # error tolerance for computing sigma; you can make this larger if it's taking too long
eta_max = 1 * DATASET_TO_SIZE[args.dataset] / 50000 # original linear scaling rule; we've arbitrarily chosen the dataset size for CIFAR as the reference. For ImageNet this will set eta_max to ~25. Note that this assumes full-batch.
t_max = 150 # feel free to expand this as large as you like if you have the resources, we didn't want to wait too long for the experiment to finish
max_r = int(eta_max * t_max)
n_runs = 3 # number of runs for each hyperparameter setting, you can increase this if you want
rtol = 0.05 # tolerance when sampling r
valid_etas = np.array([0.01] + list(np.arange(0.05, eta_max, 0.05))) # granularity of the grid is directly tied to rtol so if you make this less granular you will need to increase rtol
valid_ts = np.arange(5, t_max, 5) # similar comment as above

# Main code logic
args.max_phys_bsz = min(args.max_phys_bsz, args.batch_size)
print("GETTING DATASET")
train_loader, test_loader = get_ds(args)
args.batch_size = args.max_phys_bsz

# Function to find sigma
def compute_sigma(ts, eps_values):
    return [
        find_noise_multiplier(
            sampling_probability=1,
            num_steps=int(t),
            target_epsilon=eps,
            target_delta=args.delta,
            eps_error=eps_err,
            mu_max=5000
        ) for t, eps in zip(ts, eps_values)
    ]

# Function to launch run
def launch_config(configs, epsilon):
    for config in configs:
        run_args = copy.deepcopy(args)
        run_args.lr = config['lr']
        run_args.epochs = config['t']
        run_args.sigma = config['sigma']
        run_args.epsilon = epsilon
        print(f"Launching run with lr {config['lr']:.2f}, t {config['t']}, sigma {config['sigma']:.2f}")
        acc = launch_run(run_args, train_loader, test_loader)
        config['acc'] = acc
        searched_etas.append(config['lr'])
        searched_ts.append(config['t'])
        searched_sigmas.append(config['sigma'])
        searched_accs.append(acc)
    # sort by accuracy
    configs = sorted(configs, key=lambda x: x['acc'], reverse=True)
    print("Sorted configs for eps_1", configs)
    # get the best config
    best_config = configs[0]
    best_acc = best_config['acc']
    best_eta_1 = best_config['lr']
    best_t_1 = best_config['t']
    return best_eta_1 * best_t_1, best_acc

# Function to decompose r
def decompose_r(min_r, max_r, valid_etas, valid_ts, n_rets, rtol, n_retries=1000):
    etas, ts = [], []
    while len(etas) < n_rets:
        r = np.random.choice(np.arange(min_r, max_r, 1)) # if you have a prior on the value of r you can adjust this, an example prior is that your runs for eps_1 shouldn't use too large a value of r
        found = False
        for _ in range(n_retries): # Attempt n_retries times to find a valid decomposition
            eta = np.random.choice(valid_etas)
            t = np.random.choice(valid_ts)
            if abs(eta * t - r) <= rtol * r:
                etas.append(eta)
                ts.append(t)
                found = True
                break
        if not found:
            print(f"Failed to find a valid decomposition for r = {r}. Retrying...if this happens too much you probably need to decrease the granularity of r")
    return etas, ts

searched_etas = []
searched_ts = []
searched_sigmas = []
searched_accs = []

# First hyperparameter search
etas, ts = decompose_r(1, int(max_r//10), valid_etas, valid_ts, n_runs, rtol)
sigmas = compute_sigma(ts, [eps_1] * n_runs)
configs = [{'lr': eta, 't': t, 'sigma': sigma} for eta, t, sigma in zip(etas, ts, sigmas)]
r_1, _ = launch_config(configs, eps_1)

# Second hyperparameter search
etas, ts = decompose_r(r_1, int(max_r//2), valid_etas, valid_ts, n_runs, rtol)
sigmas = compute_sigma(ts, [eps_2] * n_runs)
configs = [{'lr': eta, 't': t, 'sigma': sigma} for eta, t, sigma in zip(etas, ts, sigmas)]
r_2, _ = launch_config(configs, eps_2)

# now compute linear interpolation 
xs = [eps_1, eps_2]
ys = [r_1, r_2]
print(f"Doing linear interpolation with eps {xs}, rs {ys}")
m, b = np.polyfit(xs, ys, 1)

# find values in valid_etas, valid_ts that are within some rtol of the extrapolated value
r_extrap = m * eps_f + b

print(f"Extrapolated value of r: {r_extrap:.2f}")
if r_extrap > max_r: # if we extrapolate outside what our prior tells us is a reasonable set of r, just use the max values
    print("Extrapolated value of r is outside of range, using max values")
    final_eta = eta_max
    final_t = t_max
else:
    final_eta, final_t = decompose_r(int(r_extrap), int(r_extrap)+1, valid_etas, valid_ts, 1, rtol, n_retries=1000)
    final_eta = final_eta[0]
    final_t = final_t[0] # Assuming decompose_r returns lists

sigma = compute_sigma([final_t], [eps_f])[0]
config = {'lr': final_eta, 't': final_t, 'sigma': sigma}
final_r, final_acc = launch_config([config], eps_f)

# Final print statements
print("Final accuracy: {}".format(final_acc))
print(f"All searched etas: {', '.join([f'{eta:.2f}' for eta in searched_etas])}")
print(f"All searched ts: {searched_ts}")
print(f"All searched sigmas: {', '.join([f'{sigma:.2f}' for sigma in searched_sigmas])}")
print(f"All searched accs: {searched_accs}")
# compute mus for hyperparameter search
mus = []
for t, sigma in zip(searched_ts, searched_sigmas):
    mu = np.sqrt(t)/sigma
    mus.append(mu)

# compute total mu
mu = np.sqrt(sum([mu**2 for mu in mus]))

mech = GaussianMechanism(1) 
# override mu 
mech.mu = mu

_, eps, _ = PRVAccountant(mech, max_self_compositions=1, eps_error=eps_err, # this computes a lower bound for eps, estimate and upper bound, and we use the estimate, you can change this to use the upper bound if you want 
                                  delta_error=1e-12).compute_epsilon(args.delta, 1)

print("Total privacy cost for final accuracy including the privacy cost of hyperparameter search ", eps)
assert eps < 1.0, "Privacy cost for hyperparameter search is too high! You've done something wrong."