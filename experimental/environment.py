import numpy as np


class PIDTuningEnvironment:

    def __init__(self, a, b, c, n, p, m, y0, horizon, noise, out_noise, n_trials):
        assert horizon > 0 and n > 0 and p > 0 and m > 0
        assert a.shape == (n, n) and b.shape == (n, p) \
               and c.shape == (m, n)
        if noise is not None:
            assert noise.shape == (n_trials, horizon, n)
        if out_noise is not None:
            assert out_noise.shape == (n_trials, horizon, m)

        self.p = p
        self.m = m
        self.y0 = y0
        self.horizon = horizon
        self.n_trials = n_trials
        self.t = None

        #original system
        self.a = a
        self.b = b
        self.c = c
        self.n = n
        self.state = None

        #noise
        self.all_noise = noise
        self.all_out_noise = out_noise
        self.noise = None
        self.out_noise = None

        self.errors = None

        self.reset(0)


    def step(self, pid_param):
        assert pid_param.ndim == 2 and pid_param.shape == (
            3, 1), 'error in action input'
        output = self.c @ self.state + self.out_noise[self.t, :].reshape(self.m, 1)
        error = self.y0 - output
        self.errors[self.t] = error
        action = self._compute_action(pid_param).reshape(self.p, 1)
        self.state = self.a @ self.state + self.b @ action + \
            self.noise[self.t, :].reshape(self.n, 1)
        self.t = self.t + 1
        return error
    

    def _compute_action(self, K):
        return K[0][0]*self.errors[self.t] + K[1][0]*np.sum(self.errors[:self.t+1]) + K[2][0]*(self.errors[self.t] - self.errors[self.t-1])
    
    
    def reset(self, i_trials):
        assert 0 <= i_trials < self.n_trials, 'trial not available'
        self.state = np.zeros((self.n, 1))
        self.t = 0
        self.noise = self.all_noise[i_trials, :, :]
        self.errors = np.zeros(self.horizon) 
        assert self.noise.ndim == 2 and self.noise.shape == (
            self.horizon, self.n), 'error in noise'
        self.out_noise = self.all_out_noise[i_trials, :, :]
        assert self.out_noise.ndim == 2 and self.out_noise.shape == (
            self.horizon, self.m), 'error in output_noise'