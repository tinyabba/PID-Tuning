import numpy as np

class Runner_opt:

    def __init__(self, environment, n_trials, horizon,
                 action_size, n_actions, actions=None):
        self.environment = environment
        self.n_trials = n_trials
        self.horizon = horizon
        self.action_size = action_size
        self.n_actions = n_actions
        self.actions = actions
        if action_size > 1:
            assert actions is not None, 'Provide actions'
            assert actions.shape == (n_actions, action_size, 1)

    def perform_simulations(self, K, idx, experiment):
        all_errors = np.zeros((self.n_trials, self.horizon))
        for sim_i in range(self.n_trials):
            self.environment.reset(sim_i)
            error_vect = self._run_simulation(K, idx, sim_i, experiment)
            assert error_vect.shape == (self.horizon,)
            all_errors[sim_i, :] = error_vect
        return all_errors

    def _run_simulation(self, K, idx, sim_i, experiment):
        optimal_errors_experiment = f"optimal_errors{experiment}.npy"
        error_vect = np.zeros(self.horizon)
        for t in range(self.horizon):
            action = K
            if self.action_size > 1:
                if isinstance(action, np.ndarray):
                    error = self.environment.step(action.reshape(
                        self.action_size, 1))
                else:
                    error = self.environment.step(self.actions[action, :
                                ].reshape(self.action_size, 1))
            else:
                error = self.environment.step(action)
            error_vect[t] = error
            if(t%100==0 or t==self.horizon-1):
                data = np.load(optimal_errors_experiment, allow_pickle=True)
                data[idx, sim_i] = error_vect
                np.save(optimal_errors_experiment, data)
        return error_vect