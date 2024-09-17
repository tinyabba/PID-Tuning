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

"""
PIDTUNING ALGORITHM, EXPERIMENT 28:
    - rho_0 < 0.4
    - 29 possible PID tuples
    - noise_sigma = 0.001
    - horizon = 20000
    - testcase 6
"""

experiment = 28
rho_0 = 0.4
testcase = 6

#Open json file
f = open(f'config/testcase_synt_{testcase}.json')
param_dict = json.load(f)

horizon = 20000
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
alg_list = [optimal, pidtuning]
errors = {alg: np.zeros((n_trials, horizon)) for alg in alg_list}

#Define noises
np.random.seed(1)
noise = np.random.normal(0, sigma, (n_trials, horizon, n))
out_noise = np.random.normal(0, sigma, (n_trials, horizon, m))



#Define range of possible PID parameters
print("Defining range of possible PID parameters")
log_space = np.logspace(0, 1, num=28, base=10)

K_P_range_start = 0.0
K_P_range_end = 1.8
K_P_range = (log_space - log_space.min()) / (log_space.max() - log_space.min()) *\
      (K_P_range_end - K_P_range_start) + K_P_range_start

K_I_range_start = 0.0
K_I_range_end = 2.3
K_I_range = (log_space - log_space.min()) / (log_space.max() - log_space.min()) *\
      (K_I_range_end - K_I_range_start) + K_I_range_start

K_D_range_start = 0.0
K_D_range_end = 0.8
K_D_range = (log_space - log_space.min()) / (log_space.max() - log_space.min()) *\
      (K_D_range_end - K_D_range_start) + K_D_range_start


#Build list of ammissible PID parameters
print("Building list of admissible PID parameters")
pid_actions = []
for K in list(itertools.product(K_P_range, K_I_range, K_D_range)):
    bar_A = utils.compute_bar_a(A, b, c, K)
    if (np.max(np.absolute(np.linalg.eigvals(bar_A))) < rho_0): 
        pid_actions.append(np.array(K).reshape(3,1))

pid_actions = np.array(pid_actions)
n_arms = pid_actions.shape[0]
print("The list of admissible PID parameters contains", n_arms, "elements")
print("Max P parameter:", np.max(pid_actions[:,0,:]))
print("Max I parameter:", np.max(pid_actions[:,1,:]))
print("Max D parameter:", np.max(pid_actions[:,2,:]))



#Upper bound for relevant quantities
action_max = pid_actions[np.argmax([np.linalg.norm(np.array(K), 2) for K in pid_actions])]
K_val = np.linalg.norm(action_max, 2)
b_val = np.linalg.norm(b, 2)
c_val = np.linalg.norm(c, 2)
spectral_rad_ub = max(np.linalg.eigvals(A))
phi_a_ub = utils.spectr(A)


#Upper bound for noise
noise_norm = []
for trial_i in range(n_trials):
    for t in range(horizon):
        noise_norm.append(np.linalg.norm(noise[trial_i, t, :]))
        noise_norm.append(np.linalg.norm(out_noise[trial_i, t, :]))
noise_ub = max(np.array(noise_norm))


#Upper bound for spectral radius of matrix bar_A
spectral_rad_list = []
for K in pid_actions:
    bar_A = utils.compute_bar_a(A, b, c, K)
    spectral_rad_list.append(np.max(np.absolute(np.linalg.eigvals(bar_A))))

spectral_rad_bar_ub = np.max(np.array(spectral_rad_list))
bar_A = utils.compute_bar_a(A, b, c, pid_actions[np.argmax(np.array(spectral_rad_list))])
phi_bar_a_ub = utils.spectr(bar_A)



#Create file for PIDTuning algorithm
pid_tuning_errors_experiment = f"pid_tuning_errors{experiment}.npy"
pulled_arms_experiment = f"pulled_arms_{experiment}.npy"
temp = np.zeros((n_trials, horizon))
np.save(pid_tuning_errors_experiment, temp)
temp = np.zeros((n_trials, horizon, 3, 1))
np.save(pulled_arms_experiment, temp)



#Running PIDTuning
agent = PIDTuningAgent(n_arms, pid_actions, horizon,
                            np.log(horizon), b_val, c_val, K_val, phi_a_ub, phi_bar_a_ub, y_0,
                            spectral_rad_ub, spectral_rad_bar_ub, noise_ub, sigma)
env = PIDTuningEnvironment(A, b, c, n, p, m, y_0, horizon, noise, out_noise, n_trials)
print('Running PID Tuning')
runner = Runner(env, agent, n_trials, horizon, 3, n_arms, pid_actions)
errors[pidtuning] = runner.perform_simulations(experiment)
np.save(pid_tuning_errors_experiment, errors[pidtuning])


