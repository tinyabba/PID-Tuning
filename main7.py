import sys
import itertools
import math
import numpy as np
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

import experimental.utils as utils
from experimental.agent import PIDTuningAgent
from experimental.environment import PIDTuningEnvironment
from experimental.runner import Runner
from experimental.runner_opt import Runner_opt

warnings.filterwarnings("ignore")

experiment = 7

#Open json file
f = open('config/testcase_synt_1.json')
param_dict = json.load(f)

horizon = param_dict['horizon']
n_trials = param_dict['n_trials']
sigma = param_dict['noise_sigma']

n = param_dict['n']
p = param_dict['p']
m = param_dict['m']

A = np.array(param_dict['A'])
b = np.array(param_dict['B'])
c = np.array(param_dict['C'])

#Step signal
y_0 = 1


#Define dictionary for the errors of the algorithms
optimal = "optimal"
pidtuning = "pidtuning"
ziegler_nichols = "ziegler_nichols"
alg_list = [optimal, pidtuning, ziegler_nichols]
errors = {alg: np.zeros((n_trials, horizon*10)) for alg in alg_list}

#Define noises
np.random.seed(1)
noise = np.random.normal(0, sigma, (n_trials, horizon*10, n))
out_noise = np.random.normal(0, sigma, (n_trials, horizon*10, m))



#Define range of possible PID parameters
log_space = np.logspace(0, 1, num=100, base=10)

K_P_range_start = 0.0
K_P_range_end = 1.5
K_P_range = (log_space - log_space.min()) / (log_space.max() - log_space.min()) *\
      (K_P_range_end - K_P_range_start) + K_P_range_start

K_I_range_start = 0.1
K_I_range_end = 2.0
K_I_range = (log_space - log_space.min()) / (log_space.max() - log_space.min()) *\
      (K_I_range_end - K_I_range_start) + K_I_range_start

K_D_range_start = 0.0
K_D_range_end = 0.8
K_D_range = (log_space - log_space.min()) / (log_space.max() - log_space.min()) *\
      (K_D_range_end - K_D_range_start) + K_D_range_start


#Build list of ammissible PID parameters
pid_actions = []
for K in list(itertools.product(K_P_range, K_I_range, K_D_range)):
    bar_A = utils.compute_bar_a(A, b, c, K)
    if (np.max(np.absolute(np.linalg.eigvals(bar_A))) < 0.3): 
        pid_actions.append(np.array(K).reshape(3,1))

pid_actions = np.array(pid_actions)
n_arms = pid_actions.shape[0]



#Run optimal algorithm
optimal_errors_experiment = f"optimal_errors{experiment}.npy"
K_opt_experiment = f"K_opt_{experiment}.npy"
K_opt_idx_experiment = f"K_opt_idx_{experiment}.npy"
all_errors_experiment = f"all_errors_{experiment}.npy"

env = PIDTuningEnvironment(A, b, c, n, p, m, y_0, horizon*10, noise, out_noise, n_trials)
print('Running Optimal algorithm')

all_errors = np.zeros((n_arms, n_trials, horizon*10))
np.save(optimal_errors_experiment, all_errors)
all_SSE = np.zeros((n_arms, n_trials))
K_opt_idx = np.zeros(n_trials)
K_opt = np.zeros((n_trials, 3, 1))
for i, K in enumerate(pid_actions):
    print("Running simulation ", i)
    runner_opt = Runner_opt(env, n_trials, horizon*10, 3, n_arms, pid_actions)
    all_errors[i] = runner_opt.perform_simulations(K, i, experiment)
    for trial_i in range(n_trials):
        all_SSE[i, trial_i] = np.sum(np.power(all_errors[i, trial_i],2))
for trial_i in range(n_trials):
    K_opt_idx[trial_i] = np.argmin(all_SSE[:, trial_i])
    K_opt[trial_i] = pid_actions[int(K_opt_idx[trial_i])]
    errors[optimal][trial_i,:] = all_errors[int(K_opt_idx[trial_i]), trial_i, :]
np.save(optimal_errors_experiment, errors[optimal])
np.save(K_opt_idx_experiment, K_opt_idx)
np.save(K_opt_experiment, K_opt)
np.save(all_errors_experiment, all_errors)




