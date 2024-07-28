import numpy as np
from tqdm.auto import tqdm


class Runner:

    def __init__(self, environment, agent, n_trials, horizon,
                 action_size, n_actions, actions=None):
        self.environment = environment
        self.agent = agent
        self.n_trials = n_trials
        self.horizon = horizon
        self.action_size = action_size
        self.n_actions = n_actions
        self.actions = actions
        if action_size > 1:
            assert actions is not None, 'Provide actions'
            assert actions.shape == (n_actions, action_size, 1)

    def perform_simulations(self):
        all_errors = np.zeros((self.n_trials, self.horizon))
        for sim_i in tqdm(range(self.n_trials)):
            print("Simulation ", sim_i)
            self.environment.reset(sim_i)
            self.agent.reset()
            error_vect = self._run_simulation(sim_i)
            assert error_vect.shape == (self.horizon,)
            all_errors[sim_i, :] = error_vect
        return all_errors

    def _run_simulation(self, sim_i):
        error_vect = np.zeros(self.horizon)
        for t in range(self.horizon):
            print("Time ", t, ", Simulation ", sim_i)
            action = self.agent.pull_arm()
            data = np.load("pulled_arms_1.npy", allow_pickle=True)
            data[sim_i, t] = action
            np.save("pulled_arms_1.npy", data)
            if self.action_size > 1:
                if isinstance(action, np.ndarray):
                    error = self.environment.step(action.reshape(
                        self.action_size, 1))
                else:
                    error = self.environment.step(self.actions[action, :
                                ].reshape(self.action_size, 1))
            else:
                error = self.environment.step(action)
            if isinstance(error, np.ndarray):
                self.agent.update(error[0, 0])
            else:
                self.agent.update(error)
            error_vect[t] = error
            if(t%100==0 or t==self.horizon-1):
                data = np.load("pid_tuning_errors1.npy", allow_pickle=True)
                data[sim_i] = error_vect
                np.save("pid_tuning_errors1.npy", data)
        return error_vect