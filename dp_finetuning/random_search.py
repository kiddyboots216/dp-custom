from utils import DATASET_TO_SIZE, parse_args, get_ds
from prv_accountant import GaussianMechanism, PRVAccountant
from prv_accountant.dpsgd import find_noise_multiplier

import numpy as np
import copy

from submit_utils import launch_run

args = parse_args()

# CONSTANTS FOR HYPERPARAMETER SEARCH
eps_1 = 1
eps_err = 0.001 # error tolerance for computing sigma; you can make this larger if it's taking too long
eta_max = 1 * DATASET_TO_SIZE[args.dataset] / 50000 # original linear scaling rule; we've arbitrarily chosen the dataset size for CIFAR as the reference. For ImageNet this will set eta_max to ~25. Note that this assumes full-batch.
t_max = 150 # feel free to expand this as large as you like if you have the resources, we didn't want to wait too long for the experiment to finish
n_runs = 10 # number of runs for each hyperparameter setting, you can increase this if you want
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

searched_etas = []
searched_ts = []
searched_sigmas = []
searched_accs = []

# First hyperparameter search
# randomly sample etas and ts from the grid
etas = np.random.choice(valid_etas, n_runs)
ts = np.random.choice(valid_ts, n_runs)
sigmas = compute_sigma(ts, [eps_1] * n_runs)
configs = [{'lr': eta, 't': t, 'sigma': sigma} for eta, t, sigma in zip(etas, ts, sigmas)]
_, _ = launch_config(configs, eps_1)

# Final print statements
print(f"All searched etas: {', '.join([f'{eta:.2f}' for eta in searched_etas])}")
print(f"All searched ts: {searched_ts}")
print(f"All searched sigmas: {', '.join([f'{sigma:.2f}' for sigma in searched_sigmas])}")
print(f"All searched accs: {searched_accs}")
print(f"Mean and standard deviation of accs: {np.mean(searched_accs):.2f} +- {np.std(searched_accs):.2f}")