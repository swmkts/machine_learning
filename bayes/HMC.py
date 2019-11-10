import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

class HMC(object):
    """
    HMC is Hamiltonian Monte Carlo sampler.
    You can get samples from distribution you assign as log_likelihood and d_loglikelihood.

    Attributes
    ----------
    log_likelihood: function
        Log likelihood of your distribution.
    param: list
        Parameter of your log likelihood function.
    d_log_likelihood: function
        Differential of log likelihood function
    tau: double
        Variance of normal distrubution.
    epsilon: double
        Time interval of leapfrog method.
    L: int
        Leapfrog step.
    iter_num: int
        Number of iteration.
    init: np.array
        Initial value of samples.

    """
    def __init__(self, log_likelihood, param, d_log_likelihood, tau, epsilon, L, iter_num, init):
        self.log_likelihood = log_likelihood
        self.param = param
        self.d_log_likelihood = d_log_likelihood
        self.tau = tau
        self.epsilon = epsilon
        self.T = T
        self.iter_num = iter_num
        self.init = init
    
    def momentum(self, p):
        """
        Calculate momentum.

        Parameters
        ----------
        p: np.array
            Kinetic energy vector. This vector's dimension must be the same of sample's dimension.
        tau: double
            Variance of kinetic energy prior.
        
        Returns
        ----------
        p.T.dot(p) / (2 * self.tau ** 2)
            Momentum.
        """
        return p.T.dot(p) / (2 * self.tau ** 2)
    
    def d_momentum(self, p):
        """
        Calculate differentiatial of momentum.

        Parameters
        ----------
        p: np.array
           Kinetic energy vector. This vector's dimension must be the same of sample's dimension.
        tau: double
            Variance of kinetic energy prior.

        Returns
        ----------
        p / self.tau**2
            Differential of momentum.
        """
        return p / self.tau**2
    
    def hamiltonian(self, x, p):
        """
        Calculate hamiltonian.
        """
        return self.momentum(p) - self.log_likelihood(x, **self.param)
    
    def leapfrog(self, x, p):
        """
        Leapfrog method

        Parameters
        ----------
        x: np.array
            Current parameter values.
        p: np.array
            Current kenetic energy.
        
        Returns
        ----------
        x: np.array
            New paramter values.
        p: np.array
            New kenetic energy.
        """
        x += - 0.5 * self.epsilon * (-1. * self.d_momentum(p))
        p += self.epsilon * self.d_log_likelihood(x, **self.param)
        x += - self.epsilon * (-1. * self.d_momentum(p))
        return x, p
    
    def proceed_HMC_iteration(self, x):
        """
        Proceed HMC searching step.

        Parameters
        ----------
        x: np.array
            Current parameter values.
        
        Returns
        ----------
        x_accepted: np.array
            Accepted parameters.
        """
        p = np.random.normal(0, self.tau, size=x.shape[0])
        p_new = p
        x_new = x
        for t in range(self.T):
            x_new, p_new = self.leapfrog(x_new, p_new)
        alpha = np.exp(self.hamiltonian(x, p) - self.hamiltonian(x_new, p_new))
        u = np.random.uniform()
        if u < alpha:
            x_accepted = np.array(x_new)
        else:
            x_accepted = x
        return x_accepted
    
    def sampling(self):
        """
        Execute HMC sampling.

        Returns
        ----------
        np.array(parameter): np.array
            Samples from the distribution you want to get.
        """
        parameter = [self.init]
        for i in range(self.iter_num):
            x_new = np.array(self.proceed_HMC_iteration(parameter[i]))
            parameter.append(x_new)
        return np.array(parameter)