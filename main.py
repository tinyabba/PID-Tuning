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


#Open json file
f = open(f'config/testcase_synt_1.json')
param_dict = json.load(f)


#Print parameters from json file
#print(f'Parameters: {param_dict}')


horizon = param_dict['horizon']
n_trials = param_dict['n_trials']
sigma = param_dict['noise_sigma']

n = param_dict['n']
p = param_dict['p']
m = param_dict['m']

A = np.array(param_dict['A'])
b = np.array(param_dict['B'])
c = np.array(param_dict['C'])


#Print system info
#utils.system_info(A, b, c)

optimal = "optimal"
pidtuning = "pidtuning"
alg_list = [optimal, pidtuning]

errors = {alg: np.zeros((n_trials, horizon)) for alg in alg_list}

#Define noises
np.random.seed(1)
noise = np.random.normal(0, sigma, (n_trials, horizon, n))
out_noise = np.random.normal(0, sigma, (n_trials, horizon, m))


#Define the range of possible PID parameters
K_P_range = np.linspace(0.01, 2.0, 10)
K_I_range = np.linspace(0.01, 1.0, 10)
K_D_range = np.linspace(0.01, 1.0, 10)


#Build list of possible PID parameters
pid_actions = []
for K in list(itertools.product(K_P_range, K_I_range, K_D_range)):
    bar_A = utils.compute_bar_a(A, b, c, K)
    if (max(np.linalg.eigvals(bar_A)) < 1):
        pid_actions.append(np.array(K).reshape(3,1))

pid_actions = np.array(pid_actions)
n_arms = pid_actions.shape[0]         


#Upper bound for relevant quantities
action_max = pid_actions[np.argmax([np.linalg.norm(np.array(K), 2) for K in pid_actions])]
K_val = np.linalg.norm(action_max, 2)
b_val = np.linalg.norm(b, 2)
c_val = np.linalg.norm(c, 2)
spectral_rad_ub = max(np.linalg.eigvals(A))
phi_a_ub = utils.spectr(A)
noise_ub = sigma*np.sqrt(2*np.log(1/0.01))
y_0 = 1


#Upper bound for spectral radius of matrix bar_A
spectral_rad_list = []
for K in pid_actions:
    bar_A = utils.compute_bar_a(A, b, c, K)
    spectral_rad_list.append(max(np.linalg.eigvals(bar_A)))

spectral_rad_bar_ub = max(np.array(spectral_rad_list))
bar_A = utils.compute_bar_a(A, b, c, pid_actions[np.argmax(np.array(spectral_rad_list))])
phi_bar_a_ub = utils.spectr(bar_A)


#Build environment
env = PIDTuningEnvironment(A, b, c, n, p, m, y_0, horizon, noise, out_noise, n_trials)


#Compute optimal PID parameters
all_errors = np.zeros((n_arms, n_trials, horizon))
all_SSE = np.zeros((n_arms, n_trials))
K_opt_idx = np.zeros(n_trials)
K_opt = np.zeros((n_trials, 3, 1))
i = 0
for K in pid_actions:
    runner_opt = Runner_opt(env, n_trials, horizon, 3, n_arms, pid_actions)
    all_errors[i] = runner_opt.perform_simulations(K)
    for trial_i in range(n_trials):
        all_SSE[i] = np.sum(np.power(all_errors[i, trial_i],2))
    i += 1
for trial_i in range(n_trials):
    K_opt_idx[trial_i] = np.argmin(all_SSE[:, trial_i])
    K_opt[trial_i] = pid_actions[int(K_opt_idx[trial_i])]
    errors[optimal][trial_i,:] = all_errors[int(K_opt_idx[trial_i]), trial_i, :]


#Running PIDTuning
agent = PIDTuningAgent(n_arms, pid_actions, horizon,
                            np.log(horizon), b_val, c_val, K_val, phi_a_ub, phi_bar_a_ub, y_0,
                            spectral_rad_ub, 0.5, noise_ub, sigma)
env = PIDTuningEnvironment(A, b, c, n, p, m, y_0, horizon, noise, out_noise, n_trials)
print('Running PID Tuning')
runner = Runner(env, agent, n_trials, horizon, 3, n_arms, pid_actions)
errors[pidtuning] = runner.perform_simulations()


#Compute regret
inst_regret = np.zeros((n_trials, horizon))
cum_regret = np.zeros((n_trials, horizon))

inst_regret = errors[optimal]**2 - errors[pidtuning]**2
cum_regret = np.cumsum(inst_regret, axis=1)
cum_regret_mean = np.mean(cum_regret, axis=0)
cum_regret_std = np.std(cum_regret, axis=0) / np.sqrt(n_trials)
print(np.shape(cum_regret_mean))

