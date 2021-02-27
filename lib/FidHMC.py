#!/usr/bin/env python3
"""
    Class for Fiducial Hamiltonian Monte Carlo using TensorFlow.

    Author: Alexander C. Murph
    Date: 2/14/21
"""
from lib.DGAWrapper import DGAWrapper
from lib.DifferentiatorDGA import DifferentiatorDGA
from lib.EvaluationFunctionWrapper import EvaluationFunctionWrapper
from lib.LogLikelihoodWrapper import LogLikelihoodWrapper
from jax import random, jit
import jax.numpy as np
import warnings
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
warnings.filterwarnings('ignore')


class FidHMC:

    def __init__(self, log_likelihood_function, dga_function, evaluation_function,
                 parameter_dimension, observed_data, lower_bounds=None, upper_bounds=None):
# =======
#         ll_wrapper = LogLikelihoodWrapper(log_likelihood_function, observed_data,
#                                           lower_bounds, upper_bounds)
#         self.ll = jit(log_likelihood_function)
        self.ll = jit(log_likelihood_function)
        self.dga_func = jit(DGAWrapper(dga_function).get_dga_function)
        self.eval_func = jit(EvaluationFunctionWrapper(evaluation_function).get_eval_function)
        self.param_dim = parameter_dimension
        self.data = observed_data
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.diff_dga = DifferentiatorDGA(self.dga_func, self.eval_func, self.param_dim, self.data)
        self.jac_l2_value = jit(self.diff_dga.calculate_fiducial_jacobian_quantity_l2)

    def _fll(self, theta):
        """
        The fiducial log likelihood.  Takes a derivative of the DGA to develop the jacobian term.
        Args:
            theta: parameter value

        Returns:
            log_sum: the log fiducial likelihood based on the data likelihood and the DGA.
        """
        transformed_theta, log_transform_jacobian = transform_parameters(theta, self.lower_bounds, self.upper_bounds)
        log_sum = np.array(self.ll(transformed_theta[0], self.data), float)
        log_sum += np.log(self.jac_l2_value(transformed_theta[0])).astype(float)
        return log_sum

    def run_NUTS(self, num_iters, burn_in, initial_value, random_key=13, step_size=2e-5):
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

        kernel = tfp.mcmc.NoUTurnSampler(self._fll, step_size=step_size)
        key, sample_key = random.split(random.PRNGKey(random_key))
        if self.lower_bounds is None or self.upper_bounds is None:
            return tfp.mcmc.sample_chain(num_iters,
                                         current_state=initial_value,
                                         kernel=kernel,
                                         trace_fn=lambda _, results: results.target_log_prob,
                                         num_burnin_steps=burn_in,
                                         seed=key)
        elif self.lower_bounds is not None and self.upper_bounds is not None:
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_value,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, pkr: pkr.log_accept_ratio,
                                                        num_burnin_steps=burn_in,
                                                        seed=key)
            new_states, log_jacobian = transform_parameters(states, self.lower_bounds, self.upper_bounds)
            return new_states, is_accepted
        else:
            raise ValueError("Please make sure your parameter bounds are properly formatted.  "
                             "At a given index, both lower and upper bounds must be equal and must be"
                             "either float type or None.")


@jit
def transform_parameters(states, lower_bounds, upper_bounds):
    """
    After drawing values from the MCMC chain that transforms to be unconstrained, transform back to the constrained
    parameter space.
    Args:
        states: the unconstrained states.

    Returns:
        c_states: the constrained states.
    """
    if len(states.shape) == 1:
        states = np.array(states).reshape(1, states.shape[0])
    new_states = np.zeros(states.shape[0])
    transform_log_jacobian = 0

    if lower_bounds is None and upper_bounds is None:
        return states, transform_log_jacobian
    else:
        for index in range(len(lower_bounds)):
            if lower_bounds[index] is None and upper_bounds[index] is None:
                new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
                                             np.array([states[:, index]], float).transpose()), axis=1)
            elif lower_bounds[index] is not None and upper_bounds[index] is not None:
                # Perform the logit transform.
                inv_logit_value = np.array(inverse_logit(states[0, index]), float)
                transform_log_jacobian += np.array(np.log((upper_bounds[index] - lower_bounds[index]) *
                                                          inv_logit_value * (1. - inv_logit_value)), float)
                logit_transform_states = np.array([lower_bounds[index] +
                                                   (upper_bounds[index] - lower_bounds[index]) *
                                                   inverse_logit(states[:, index])], float)
                new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
                                             logit_transform_states.transpose()), axis=1)
            elif upper_bounds[index] is None:
                # Perform the lower logarithmic transform.
                transform_log_jacobian += np.array(states[0, index], float)
                log_transform_states = np.array([np.exp(states[:, index]) + lower_bounds[index]], float)
                new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
                                             log_transform_states.transpose()), axis=1)
            elif lower_bounds[index] is None:
                # Perform the upper logarithmic transform.
                transform_log_jacobian += np.array(states[0, index], float)
                log_transform_states = np.array([lower_bounds[index] - np.exp(states[:, index])], float)
                new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
                                             log_transform_states.transpose()), axis=1)
            else:
                raise TypeError("Please make sure your parameter bounds are type float or None.")
    return new_states[:, 1:], transform_log_jacobian


@jit
def inverse_logit(value):
    return (1. + np.exp(-value)) ** (-1.)
