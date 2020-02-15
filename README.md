# machine_learning

* Implementation of machine learning algorithm and related things.

* Mainly using Numpy.

## Directories

[bayes](https://github.com/swmkts/machine_learning/tree/master/bayes)

[monte_carlo](https://github.com/swmkts/machine_learning/tree/master/monte_carlo)

[optimization](https://github.com/swmkts/machine_learning/tree/master/optimization)

## bayes

* Implementation of some bayesian learning method.

  * [Gaussian process](https://github.com/swmkts/machine_learning/blob/master/bayes/GaussianProcess.ipynb)
  
  * [Dirichlet process, Pitman-Yor process and stick breaking process](https://github.com/swmkts/machine_learning/blob/master/bayes/CRP_PitmanYor_SBP.ipynb)
  
  * [Hamiltonian Monte Carlo](https://github.com/swmkts/machine_learning/blob/master/bayes/HamiltonianMonteCarlo.ipynb)
    * This contains below.
    * Linear regression by HMC
    * Logistic regression by HMC
  
  * [Logistic regression by VB](https://github.com/swmkts/Bayesian_logistic_regression/blob/master/input/.ipynb_checkpoints/VB_Logistic_Regression-checkpoint.ipynb)
  
  * [Bayesian Hidden Markov Model](https://github.com/swmkts/machine_learning/blob/master/bayes/BayesianHMM.ipynb)

  * [Shrinkage factor](https://github.com/swmkts/machine_learning/blob/master/bayes/shrinkage_factor.ipynb)
    * This is related to bayesian regularization method.
    * The shape of prior distribution of parameters.
  
  

## monte carlo method

* Basics of monte carlo method for bayesian learning.

* This contains below.
  * Sampling method
    * inversion method
    * rejection method
    * ratio of uniforms method
      * Sampling from truncated normal distribution
  * Monte carlo integral
    * self-Normalized inportance sampling
  * Markov chain monte carlo(MCMC) method
    * Metropolis hastings
  * Hamiltonian Monte Carlo method
    * This is in [Hamiltonian Monte Carlo](https://github.com/swmkts/machine_learning/blob/master/bayes/HamiltonianMonteCarlo.ipynb).


## optimization
* [backtrack line search](https://github.com/swmkts/machine_learning/blob/master/optimization/backtracking_line_search.ipynb)
* [method of multipliers](https://github.com/swmkts/machine_learning/blob/master/optimization/dual_ascent_multipliers.ipynb)
 