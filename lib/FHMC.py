#!/usr/bin/env python3
"""
    Class for Fiducial Hamiltonian Monte Carlo using TensorFlow.

    Author: Alexander C. Murph
    Date: 2/14/21
"""
from jax import random
import jax.numpy as np
import tensorflow_probability as tfp
from lib.DifferentiatorDGA import DifferentiatorDGA
tfp = tfp.substrates.jax


class FHMC:

    def __init__(self, log_likelihood_function, dga_function, evaluation_function, parameter_dimension, observed_data):
        self.ll = log_likelihood_function
        self.dga_func = dga_function
        self.eval_func = evaluation_function
        self.param_dim = parameter_dimension
        self.data = observed_data

    def _fll(self, theta):
        """
        The fiducial log likelihood.  Takes a derivative of the DGA to develop the jacobian term.
        Args:
            theta: parameter value

        Returns:
            log_sum: the log fiducial likelihood based on the data likelihood and the DGA.
        """
        log_sum = self.ll(theta, self.data)
        diff_dga = DifferentiatorDGA(self.dga_func, self.eval_func, self.param_dim, self.data)
        log_sum += np.log(diff_dga.calculate_fiducial_jacobian_quantity_l2(theta))
        return log_sum

    def run_NUTS(self, num_iters, burn_in, initial_value, random_key=13, step_size=1e-3):
        """
        Method to perform a No-U-Turn sampler for a target fiducial density.  Uses the well-maintained
        functionalities in TensorFlow and JAX.
        Args:
            num_iters: Total number of iterations to take the algorithm.
            burn_in: Number of samples to burn when warming up the algorithm.
            initial_value: A starting value for the MCMC chain.
            random_key: Optional, sets the random seed.
            step_size: Optional, sets the step size for the hamiltonian step.

        Returns:
            states: the parameter samples drawn from the MCMC chain.
            log_probs: the log probability values of the fiducial density at each iteration.
        """
        assert num_iters > burn_in, "You are burning more samples than you are taking."

        kernel = tfp.mcmc.NoUTurnSampler(self._fll, step_size=step_size)
        key, sample_key = random.split(random.PRNGKey(random_key))
        return tfp.mcmc.sample_chain(num_iters,
                                     current_state=initial_value,
                                     kernel=kernel,
                                     trace_fn=lambda _, results: results.target_log_prob,
                                     num_burnin_steps=burn_in,
                                     seed=key)
