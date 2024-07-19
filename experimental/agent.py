from abc import ABC, abstractmethod
import numpy as np
import math
import itertools
from random import Random


class Agent(ABC):
    def __init__(self, n_arms, random_state=1):
        self.n_arms = n_arms
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.last_pull = None
        np.random.seed(random_state)
        self.randgen = Random(random_state)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, reward):
        pass

    def reset(self):
        self.t = 0
        self.last_pull = None




class PIDTuningAgent(Agent):
    def __init__(self, n_arms, actions, horizon, lmbd, b_ub, c_ub, K_ub, 
                 phi_a_ub, phi_a_bar_ub, y0_ub, spectral_rad_ub, spectral_rad_bar_ub, noise_ub,
                 sigma,  epsilon=0.000001, random_state=1):
        
        super().__init__(n_arms, random_state)

        spectral_rad_ub = np.abs(spectral_rad_ub)
        spectral_rad_bar_ub = np.abs(spectral_rad_bar_ub)
        assert lmbd > 0 and spectral_rad_ub < 1 and spectral_rad_bar_ub < 1
        assert actions.shape == (n_arms, 3, 1)
        self.H = np.log(horizon)/np.log(1/spectral_rad_bar_ub)
        self.powers, self.features_dim = self._features_mapping_dimension()
        self.features_dim += 1
        self.actions = actions
        self.horizon = horizon
        self.lmbd = lmbd

        #Upper bounds for relevant quantities in the original system
        self.spectral_rad_ub = spectral_rad_ub
        self.phi_a_ub = phi_a_ub
        self.b_ub = b_ub
        self.c_ub = c_ub
        self.y0_ub = y0_ub
        self.K_ub = K_ub
        self.noise_ub = noise_ub
        self.sigma = sigma

        #Upper bounds for relevant quantities in the new system
        self.spectral_rad_bar_ub = spectral_rad_bar_ub
        self.phi_a_bar_ub = phi_a_bar_ub
        self.b_bar_ub = np.sqrt(K_ub**2*b_ub**2 + 2)
        self.c_bar_ub = c_ub
        self.theta_vect_ub = self.c_bar_ub**(self.H) * self.b_bar_ub**(self.H) *\
                             self.phi_a_bar_ub * self.spectral_rad_bar_ub**(self.H)               
        self.features_vect_ub = np.sqrt(1 + self.features_dim*K_ub**(4*self.H))
        self.x_bar_ub = (self.b_bar_ub*self.y0_ub + self.b_bar_ub*self.noise_ub +\
                          self.noise_ub)/(1 - self.spectral_rad_bar_ub)
        self.noise_xi_ub = 2*self.c_bar_ub**2*self.x_bar_ub**2*phi_a_bar_ub**2 +\
                           4*self.c_bar_ub*self.x_bar_ub*y0_ub*phi_a_bar_ub +\
                           2*self.c_bar_ub*phi_a_bar_ub*noise_ub*self.x_bar_ub +\
                           2*y0_ub**2 + 2*y0_ub*noise_ub + noise_ub**2 + sigma**2
        
        self.last_mapping = None

        if spectral_rad_ub < epsilon:
            self.t_estimate = np.linspace(0, horizon - 1, horizon, dtype=int)
        else:
            self.t_estimate = []
            t_estimate = math.floor(self.H)
            m = 2            
            while(t_estimate < self.horizon):
                self.t_estimate.append(t_estimate)
                t_estimate = m*math.floor(self.H)
                m += 1
            self.t_estimate.append(0)
        self.t_newaction = list(np.array(self.t_estimate) + 1)           

        self.reset()
        

    def reset(self):
        super().reset()
        self.V_t = np.array(self.lmbd * np.eye(self.features_dim), dtype=np.float32)            
        self.b_vect = np.array(np.zeros((self.features_dim, 1)) , dtype=np.float32)            
        self.hat_theta_vect = np.array(np.zeros((self.features_dim, 1)), dtype=np.float32)   
        self.first = True
        self.newaction_vect_idx = 0
        self.estimate_vect_idx = 0
        return self
    
    
    def pull_arm(self):
        if self.first:
            print("pull first arm")
            K_t = self.actions[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = K_t.reshape(3, 1)
            self.last_mapping = self._features_mapping(self.last_pull)
            self.first = False
        elif self.t == self.t_newaction[self.newaction_vect_idx]:
            print("pull arm")
            K_t, _ = self._estimate_pidtuning_action()
            self.last_pull = K_t.reshape(3, 1)
            self.last_mapping = self._features_mapping(self.last_pull)
            self.newaction_vect_idx += 1
        return self.last_pull
    

    def _features_mapping_dimension(self):
        powers = []
        for h in range(1, 2*math.floor(self.H)+1):
            for elem in list(itertools.product(range(h,-1,-1), range(h,-1,-1), range(h,-1,-1))):
                if ((elem[0] + elem[1] + elem[2]) == h):
                    powers.append(elem)
        return powers, len(powers)  


    def _features_mapping(self, K):   
        mapping = [1]
        for p in self.powers:
            mapping.append(K[0][0]**p[0] * K[1][0]**p[1] * K[2][0]**p[2])

        mapping = np.array(mapping, dtype=np.float32)
        mapping = mapping.reshape(mapping.shape[0], 1)
        return mapping      


    def update(self, error):
        if self.t == self.t_estimate[self.estimate_vect_idx]:
            print("update")
            self.V_t = self.V_t + (self.last_mapping @ self.last_mapping.T)
            self.b_vect = self.b_vect + self.last_mapping * error**2
            self.hat_theta_vect = np.linalg.inv(self.V_t) @ self.b_vect
            self.estimate_vect_idx += 1
        self.t += 1

    
    def _beta_t_fun_pidtuning(self):
        return self.theta_vect_ub * np.sqrt(self.lmbd) + \
               np.sqrt(
                   2 * self.noise_xi_ub**2 * (
                           np.log(self.horizon) + (self.features_dim / 2) *
                           np.log(1 + (self.t * (self.features_vect_ub ** 2)) / (
                                   self.features_dim * self.lmbd))
                   )
               )
    

    def _estimate_pidtuning_action(self):
        print("estimate action")
        bound = self._beta_t_fun_pidtuning()
        obj_vals = np.array(np.zeros(self.n_arms), dtype=np.float32)
        for i, act_i in enumerate(self.actions):
            act_i = act_i.reshape(3, 1)
            obj_vals[i] = self.hat_theta_vect.T @ self._features_mapping(act_i) - bound * np.sqrt(
                self._features_mapping(act_i).T @ np.linalg.inv(self.V_t) @ self._features_mapping(act_i))
        return self.actions[np.argmin(obj_vals), :], np.argmin(obj_vals)