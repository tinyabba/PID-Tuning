import numpy as np

class RunnerStep:

    def __init__(self, environment, n_trials, horizon):
        self.environment = environment
        self.n_trials = n_trials
        self.horizon = horizon

    def perform_simulations(self, y0, experiment):
        all_outputs = np.zeros((self.n_trials, self.horizon))
        y0 = np.array([y0])
        y0 = y0.reshape(1,1)
        for sim_i in range(self.n_trials):
            self.environment.reset(sim_i)
            all_outputs[sim_i, :] = self._run_simulation(sim_i, y0, experiment)
        return all_outputs

    def _run_simulation(self, sim_i, y0, experiment):
        filename = f"step_{experiment}.npz"
        outputs_vect = np.zeros(self.horizon)

        # Load the data once outside the loop
        loaded = np.load(filename, allow_pickle=True)
        data = loaded['all_outputs']
        
        for t in range(self.horizon):
            output = self.environment.step(y0)
            outputs_vect[t] = output

            # Save data at intervals and at the end
            if (t % 10000 == 0 or t == self.horizon - 1):
                print("Time", t, ", Simulation", sim_i)
                data[sim_i] = outputs_vect
                np.savez_compressed(filename, all_outputs = data)
        
        return outputs_vect
