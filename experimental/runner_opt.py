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
            all_errors[sim_i, :] = self._run_simulation(K, idx, sim_i, experiment)
        return all_errors

    def _run_simulation(self, K, idx, sim_i, experiment):
        filename = f"experiment_{experiment}.npz"
        error_vect = np.zeros(self.horizon)
        
        # Determine the action type and prepare data loading
        if self.action_size > 1:
            if isinstance(K, np.ndarray):
                action = K.reshape(self.action_size, 1)
                action_is_array = True
            else:
                action_is_array = False

        # Load the data once outside the loop
        loaded = np.load(filename, allow_pickle=True)
        data = loaded['all_errors']
        pid_actions = loaded['pid_actions']
        
        for t in range(self.horizon):
            if self.action_size > 1:
                if action_is_array:
                    error = self.environment.step(action)
                else:
                    error = self.environment.step(self.actions[K].reshape(self.action_size, 1))
            else:
                error = self.environment.step(K)

            error_vect[t] = error

            # Save data at intervals and at the end
            if (t % 100 == 0 or t == self.horizon - 1):
                data[idx, sim_i] = error_vect
                np.savez_compressed(filename, all_errors = data, pid_actions=pid_actions)
        
        return error_vect
