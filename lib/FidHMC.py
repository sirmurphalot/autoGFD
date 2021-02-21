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
        self.ll = jit(LogLikelihoodWrapper(log_likelihood_function, observed_data,
                                           lower_bounds, upper_bounds).get_log_likelihood)
        self.dga_func = jit(DGAWrapper(dga_function).get_dga_function)
        self.eval_func = jit(EvaluationFunctionWrapper(evaluation_function).get_eval_function)
        self.param_dim = parameter_dimension
        self.data = observed_data
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.diff_dga = DifferentiatorDGA(self.dga_func, self.eval_func, self.param_dim, self.data)

    def _fll(self, theta):
        """
        The fiducial log likelihood.  Takes a derivative of the DGA to develop the jacobian term.
        Args:
            theta: parameter value

        Returns:
            log_sum: the log fiducial likelihood based on the data likelihood and the DGA.
        """
        log_sum = self.ll(theta)
        log_sum += np.log(self.diff_dga.calculate_fiducial_jacobian_quantity_l2(theta)).astype(float)
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
        if self.lower_bounds is None or self.upper_bounds is None:
            return tfp.mcmc.sample_chain(num_iters,
                                         current_state=initial_value,
                                         kernel=kernel,
                                         trace_fn=lambda _, results: results.target_log_prob,
                                         num_burnin_steps=burn_in,
                                         seed=key)
        elif self.lower_bounds is not None and self.upper_bounds is not None:
            states, log_probs = tfp.mcmc.sample_chain(num_iters,
                                                      current_state=initial_value,
                                                      kernel=kernel,
                                                      trace_fn=lambda _, results: results.target_log_prob,
                                                      num_burnin_steps=burn_in,
                                                      seed=key)
            new_states = np.zeros(states.shape[0])
            for index in range(self.param_dim):
                if self.lower_bounds[index] is None or self.upper_bounds[index] is None:
                    new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
                                                 np.asarray([states[:, index]]).transpose()), axis=1)
                elif type(self.lower_bounds[index]) is float and type(self.upper_bounds[index]) is float:
                    logit_states = np.array([(states[:, index] -
                                              self.lower_bounds[index]) * ((self.upper_bounds[index] -
                                                                            self.lower_bounds[index]) ** (-1))], float)
                    new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
                                                 logit_states.transpose()), axis=1)
                else:
                    raise TypeError("Please make sure your parameter bounds are type float or None.")
            new_states = new_states[:, 1:]
            return new_states, log_probs
        else:
            raise ValueError("Please make sure your parameter bounds are properly formatted.  "
                             "At a given index, both lower and upper bounds must be equal and must be"
                             "either float type or None.")


# @jit
# def logit_transform_states(states, lower_bounds, upper_bounds):
#     new_states = np.zeros(states.shape[0])
#     print(lower_bounds[1])
#     print(upper_bounds[1])
#     for index in range(states.shape[1]):
#         if lower_bounds[index] is None or upper_bounds[index] is None:
#             new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
#                                          np.asarray([states[:, index]]).transpose()), axis=1)
#         elif type(lower_bounds[index]) is float and type(upper_bounds[index]) is float:
#             logit_states = np.array([(states[:, index] -
#                                       lower_bounds[index]) * ((upper_bounds[index] -
#                                                                lower_bounds[index]) ** (-1))], float)
#             new_states = np.concatenate((new_states.reshape(states.shape[0], index + 1),
#                                          logit_states.transpose()), axis=1)
#         else:
#             raise TypeError("Please make sure your parameter bounds are type float or None.")
#     return new_states[:, 1:]
