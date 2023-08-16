from utils import DATASET_TO_SIZE, parse_args, get_ds
from prv_accountant import GaussianMechanism, PRVAccountant
from prv_accountant.dpsgd import find_noise_multiplier

import numpy as np
import copy

from submit_utils import launch_run

args = parse_args()

args.max_phys_bsz = min(args.max_phys_bsz, args.batch_size)
n_accum_steps = max(1, np.ceil(args.batch_size / args.max_phys_bsz))
print("GETTING DATASET")
train_loader, test_loader = get_ds(args)
args.batch_size = args.max_phys_bsz

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
eps_err = 0.001 # error tolerance for computing sigma; you can make this larger if it's taking too long

eta_max = 1 * DATASET_TO_SIZE[args.dataset] / 50000 # original linear scaling rule; we've arbitrarily chosen the dataset size for CIFAR as the reference. For ImageNet this will set eta_max to ~25. Note that this assumes full-batch.
t_max = 150 # feel free to expand this as large as you like if you have the resources, we didn't want to wait too long for the experiment to finish
max_r = int(eta_max * t_max)
n_runs = 3 # number of runs for each hyperparameter setting, you can increase this if you want
rtol = 0.05 # tolerance when sampling r

valid_etas = np.array([0.01] + list(np.arange(0.05, eta_max, 0.05))) # granularity of the grid is directly tied to rtol so if you make this less granular you will need to increase rtol
valid_ts = np.arange(5, t_max, 5) # similar comment as above

searched_etas = []
searched_ts = []
searched_sigmas = []
searched_accs = []

# first hparam search
etas = []
ts = []

while len(etas) < n_runs:
    r = np.random.choice(np.arange(1, int(max_r//10), 1)) # if you have a prior on the value of r you can adjust this, an example prior is that your runs for eps_1 shouldn't use too large a value of r
    found = False
    for _ in range(1000): # Attempt 1000 times to find a valid decomposition
        eta = np.random.choice(valid_etas)
        t = np.random.choice(valid_ts)
        if abs(eta * t - r) <= rtol * r:
            etas.append(eta)
            ts.append(t)
            found = True
            break
    if not found:
        print(f"Failed to find a valid decomposition for r = {r}. Retrying...if this happens too much you probably need to decrease the granularity of r")

sigmas = []

# compute sigmas for corresponding ts assuming eps_1 = 0.01 (modify as necessary)
eps_minis = [eps_1] * n_runs
for t, eps in zip(ts, eps_minis):
    sigma = find_noise_multiplier(
            sampling_probability=1,
            num_steps=int(t),
            target_epsilon=eps,
            target_delta=args.delta, # don't worry about this, we're not doing basic composition anyways
            eps_error=eps_err,
            mu_max=5000)
    sigmas.append(sigma)

configs = [{
    'lr': eta,
    't': t,
    'sigma': sigma,
} for eta, t, sigma in zip(etas, ts, sigmas)]

for config in configs:
    run_args = copy.deepcopy(args)
    run_args.lr = config['lr']
    run_args.epochs = config['t']
    run_args.sigma = config['sigma']
    run_args.epsilon = eps_1
    print("Launching run with lr {}, t {}, sigma {}".format(config['lr'], config['t'], config['sigma']))
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
best_eta_1 = best_config['lr']
best_t_1 = best_config['t']

valid_etas = np.arange(best_eta_1, eta_max, 0.05)
valid_ts = np.arange(best_t_1, t_max, 5)

etas = []
ts = []
min_r = int(best_eta_1 * best_t_1)
while len(etas) < n_runs:
    r = np.random.choice(np.arange(min_r, int(max_r//2), 1)) # if you have a prior on the value of r you can adjust this, an example prior is that your runs for eps_2 shouldn't use too large a value of r
    found = False
    for _ in range(1000): # Attempt 1000 times to find a valid decomposition
        eta = np.random.choice(valid_etas)
        t = np.random.choice(valid_ts)
        if abs(eta * t - r) <= rtol * r:
            etas.append(eta)
            ts.append(t)
            found = True
            break
    if not found:
        print(f"Failed to find a valid decomposition for r = {r}. Retrying...if this happens too much you probably need to decrease the granularity of r")

sigmas = []


eps_minis = [eps_2] * n_runs
for t, eps in zip(ts, eps_minis):
    sigma = find_noise_multiplier(
            sampling_probability=1,
            num_steps=int(t),
            target_epsilon=eps,
            target_delta=args.delta, # don't worry about this, we're not doing basic composition anyways
            eps_error=eps_err,
            mu_max=5000)
    sigmas.append(sigma)

configs = [{
    'lr': eta,
    't': t,
    'sigma': sigma,
} for eta, t, sigma in zip(etas, ts, sigmas)]

for config in configs:
    run_args = copy.deepcopy(args)
    run_args.lr = config['lr']
    run_args.epochs = config['t']
    run_args.sigma = config['sigma']
    run_args.epsilon = eps_2
    print("Launching run with lr {}, t {}, sigma {}".format(config['lr'], config['t'], config['sigma']))
    acc = launch_run(run_args, train_loader, test_loader) 
    config['acc'] = acc
    searched_etas.append(config['lr'])
    searched_ts.append(config['t'])
    searched_sigmas.append(config['sigma'])
    searched_accs.append(acc)

# sort by accuracy
configs = sorted(configs, key=lambda x: x['acc'], reverse=True)
print("Sorted configs for eps_2", configs)
# get the best config
best_config = configs[0]
best_eta_2 = best_config['lr']
best_t_2 = best_config['t']

# now compute linear interpolation 
xs = [eps_1, eps_2]
ys = [best_eta_1 * best_t_1, best_eta_2 * best_t_2]
print("Doing linear interpolation with eps {}, rs {}".format(xs, ys))
m, b = np.polyfit(xs, ys, 1)

# find values in valid_etas, valid_ts that are within some rtol of the extrapolated value
r_extrap = m * eps_f + b

print("Extrapolated value of r: {}".format(r_extrap))
if r_extrap > max_r: # this generally won't happen but just in case
    print("Extrapolated value of r is outside of range, using max values")
    final_eta = eta_max
    final_t = t_max
else:
    final_eta = None
    final_t = None

    valid_etas = np.arange(best_eta_2, eta_max, 0.05)
    valid_ts = np.arange(best_t_2, t_max, 5)

    for i in range(1000):
        eta = np.random.choice(valid_etas)
        t = np.random.choice(valid_ts)
        if abs(eta * t - r_extrap) <= rtol * r_extrap:
            final_eta = eta
            final_t = t
            break
    if final_eta is None:
        print('failed to find extrapolated value, increasing rtol')
    for i in range(1000):
        eta = np.random.choice(valid_etas)
        t = np.random.choice(valid_ts)
        if abs(eta * t - r_extrap) <= rtol * 2 * r_extrap:
            final_eta = eta
            final_t = t
            break
    if final_eta is None:
        print('failed to find extrapolated value; there is an issue!')
        raise ValueError

config = {
    'lr': final_eta,
    't': final_t,
    'sigma': find_noise_multiplier(
            sampling_probability=1,
            num_steps=int(final_t),
            target_epsilon=eps_f,
            target_delta=args.delta, # don't worry about this, we're not doing basic composition anyways
            eps_error=eps_err,
    mu_max=5000)
}
args.lr = config['lr']
args.epochs = config['t']
args.sigma = config['sigma']

print("Launching run with lr {}, t {}, sigma {}".format(config['lr'], config['t'], config['sigma']))
final_acc = launch_run(args, train_loader, test_loader)

print("Final accuracy: {}".format(final_acc))
searched_accs.append(final_acc)
searched_etas.append(config['lr'])
searched_ts.append(config['t'])
searched_sigmas.append(config['sigma'])
print("All searched etas:", ', '.join([f"{eta:.2f}" for eta in searched_etas]))
print("All searched ts:", searched_ts)
print("All searched sigmas:", ', '.join([f"{sigma:.3f}" for sigma in searched_sigmas]))
print("All searched accs:", searched_accs)
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