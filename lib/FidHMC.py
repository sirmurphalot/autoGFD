#!/usr/bin/env python3
"""
    Class for Fiducial Hamiltonian Monte Carlo using TensorFlow.

    Author: Alexander C. Murph
    Date: 2/14/21
"""
from lib.DGAWrapper import DGAWrapper
from lib.DifferentiatorDGA import DifferentiatorDGA
from lib.EvaluationFunctionWrapper import EvaluationFunctionWrapper
from jax import random, jit
import jax.numpy as np
import warnings
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
warnings.filterwarnings('ignore')


class FidHMC:
    def __init__(self, log_likelihood_function, dga_function, evaluation_function,
                 parameter_dimension, observed_data, lower_bounds=None, upper_bounds=None,
                 number_of_cores=1, user_l2_jac_det_term=None):
        """
        Main class to perform fiducial inference using the JAX autodifferentiator.
        Args:
            log_likelihood_function: User defined python function.  Describes the data distribution.
            dga_function: User defined python function.  Describes relationship between parameters,
                            data, and random quantity in the form Data = F(Parameters, Random Quantity).
            evaluation_function: User defined python function.  Describes relationship between parameters,
                            data, and random quantity in the form Random Quantity = F^-1(Data, Random Quantity).
            parameter_dimension: Integer. The number of parameters to be learned.
            observed_data: Vector or Matrix.  The observed data, assumed to be iid.
            lower_bounds: Optional Vector.  Lower bounds of parameters, if they are left censored.
            upper_bounds: Optional Vector.  Upper bounds of parameters, if they are right censored.
            number_of_cores: Optional Integer.  Number of cores available for parallelization.
            user_l2_jac_det_term: Optional python function.  If one is unable to express their DGA in a way that can
                                    be used by JAX, one can also calculate the derivative directly and input it here.
        """
        self.ll = jit(log_likelihood_function)
        self.param_dim = parameter_dimension
        self.data = observed_data
        self.user_l2_jac_det_term = user_l2_jac_det_term
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        if dga_function is not None:
            self.dga_func = jit(DGAWrapper(dga_function).get_dga_function)
            self.eval_func = jit(EvaluationFunctionWrapper(evaluation_function).get_eval_function)
            self.diff_dga = DifferentiatorDGA(self.dga_func, self.eval_func, self.param_dim, self.data)
            self.jac_l2_value = self.diff_dga.calculate_fiducial_jacobian_quantity_l2
        else:
            self.jac_l2_value = self._jac_l2_wrapper
        self.number_of_cores = number_of_cores

    def _jac_l2_wrapper(self, theta):
        return self.user_l2_jac_det_term(theta, self.data)

    def _fll(self, theta):
        """
        The fiducial log likelihood.  Takes a derivative of the DGA to develop the jacobian term.
        Args:
            theta: parameter value

        Returns:
            log_sum: the log fiducial likelihood based on the data likelihood and the DGA.
        """
        transformed_theta, log_transform_jacobian = transform_parameters(theta, self.lower_bounds,
                                                                         self.upper_bounds)
        log_sum = np.array([self.ll(transformed_theta[0, i, :],
                                    self.data) for i in range(transformed_theta.shape[1])])
        log_sum += np.add(log_sum,
                          np.array([np.log(self.jac_l2_value(transformed_theta[0, i,
                                                             :])) for i in range(transformed_theta.shape[1])]))
        return log_sum.astype(float)

    def run_NUTS(self, num_iters, burn_in, initial_value, random_key=13, step_size=2e-5, num_chains=1):
        """
        Method to perform a No-U-Turn sampler for a target fiducial density.  Uses the well-maintained
        functionalities in TensorFlow and JAX.
        Args:
            num_chains: Integer. Number of independent MCMC chains to create.
            num_iters: Integer. Total number of iterations to take the algorithm.
            burn_in: Integer. Number of samples to burn when warming up the algorithm.
            initial_value: Vector. A starting value for the MCMC chain, must be the same dimension as the parameter.
            random_key: Optional Integer.  Sets the random seed.
            step_size: Optional Double.  Sets the step size for the hamiltonian step.

        Returns:
            states: the parameter samples drawn from the MCMC chain.
            is_accept: the log acceptance ratio.
        """
        print("---------------------------------")
        print("Creating the NUTS Kernel...")
        nuts_kernel = tfp.mcmc.NoUTurnSampler(self._fll, step_size=step_size)
        print("Adapting Step Size...")
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            nuts_kernel,
            num_adaptation_steps=int(burn_in * 0.8),
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )
        key, sample_key = random.split(random.PRNGKey(random_key))
        # Set up the Number of Independent Chains requested:
        initial_values = []
        for index in range(num_chains):
            initial_values.append(initial_value)
        initial_values = np.stack(initial_values)

        if self.lower_bounds is None or self.upper_bounds is None:
            print("Drawing from the Markov Chain...")
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_values,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
                                                        num_burnin_steps=burn_in,
                                                        seed=key,
                                                        parallel_iterations=self.number_of_cores)
            print("---------------------------------")
            states = np.concatenate(states)
            return states, is_accepted
        elif self.lower_bounds is not None and self.upper_bounds is not None:
            print("Drawing from the Markov Chain...")
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_values,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
                                                        num_burnin_steps=burn_in,
                                                        seed=key,
                                                        parallel_iterations=self.number_of_cores)
            new_states, log_jacobian = transform_parameters(states, self.lower_bounds, self.upper_bounds)
            new_states = np.concatenate(new_states)
            print("---------------------------------")
            return new_states, is_accepted
        else:
            raise ValueError("Please make sure your parameter bounds are properly formatted.  "
                             "At a given index, both lower and upper bounds must be equal and must be"
                             "either float type or None.")

    def run_HMC(self, num_iters, burn_in, initial_value, random_key=13, step_size=2e-5, num_chains=1):
        """
        Method to perform a Hamiltonian Monte Carlo sampler for a target fiducial density.  Uses the well-maintained
        functionalities in TensorFlow and JAX.
        Args:
            num_chains: Integer. Number of independent chains to run.
            num_iters: Integer. Total number of iterations to take the algorithm.
            burn_in: Integer. Number of samples to burn when warming up the algorithm.
            initial_value: Vector. A starting value for the MCMC chain, must be the same dimension as the parameter.
            random_key: Optional Integer. sets the random seed.
            step_size: Optional Double.  Sets the step size for the hamiltonian step.

        Returns:
            states: the parameter samples drawn from the MCMC chain.
            is_accepted: the log acceptance ratio.
        """
        print("---------------------------------")
        print("Creating the HMC Kernel...")
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self._fll,
            num_leapfrog_steps=2,
            step_size=step_size)
        print("Adapting Step Size...")
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=int(burn_in * 0.8)
        )
        key, sample_key = random.split(random.PRNGKey(random_key))
        # Set up the Number of Independent Chains requested:
        initial_values = []
        for index in range(num_chains):
            initial_values.append(initial_value)
        initial_values = np.stack(initial_values)

        if self.lower_bounds is None or self.upper_bounds is None:
            print("Drawing from the Markov Chain...")
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_values,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
                                                        num_burnin_steps=burn_in,
                                                        seed=key,
                                                        parallel_iterations=self.number_of_cores)
            print("---------------------------------")
            states = np.concatenate(states)
            return states, is_accepted
        elif self.lower_bounds is not None and self.upper_bounds is not None:
            print("Drawing from the Markov Chain...")
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_values,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
                                                        num_burnin_steps=burn_in,
                                                        seed=key,
                                                        parallel_iterations=self.number_of_cores)
            new_states, log_jacobian = transform_parameters(states, self.lower_bounds, self.upper_bounds)
            print("---------------------------------")
            new_states = np.concatenate(new_states)
            return new_states, is_accepted
        else:
            raise ValueError("Please make sure your parameter bounds are properly formatted.  "
                             "At a given index, both lower and upper bounds must be equal and must be"
                             "either float type or None.")

    def run_RWM(self, num_iters, burn_in, initial_value, random_key=13, proposal_scale=0.3, num_chains=1):
        """
        Run a random-walk Metropolis-Hastings Markov Chain Monte Carlo algorithm on the fiducial log likelihood.
        Currently uses a Cauchy proposal distribution.  Does NOT currently have a scale-tuning mechanism.
        I may hard-code this method to implement such a tuning mechanism.
        Uses the well-maintained functionalities in TensorFlow and JAX.
        Args:
            num_chains: Integer. Number of independent chains to run.
            num_iters: Integer. Number of MCMC draws requested from the kernel.
            burn_in: Integer. Number of burn in steps to warm up the kernel.
            initial_value: Vector. A starting value for the MCMC chain, must be the same dimension as the parameter.
            random_key: Optional Integer. A random seed.
            proposal_scale: Optional Double.  This is not yet automatically tuned, so it is highly
                                suggested that a user set and experiment with this.

        Returns:
            states: 'num_iters' draws from the MCMC chain.
            is_accept: the log acceptance ratio.
        """

        def cauchy_new_state_fn(scale, dtype):
            cauchy = tfd.Cauchy(loc=dtype(0), scale=dtype(scale))

            def _fn(state_parts, seed):
                next_state_parts = []
                part_seeds = tfp.random.split_seed(
                    seed, n=len(state_parts), salt='rwmcauchy')
                for sp, ps in zip(state_parts, part_seeds):
                    next_state_parts.append(sp + cauchy.sample(
                        sample_shape=sp.shape, seed=ps))
                return next_state_parts

            return _fn

        print("---------------------------------")
        print("Creating the Metropolis-Hastings Kernel...")
        kernel = tfp.mcmc.RandomWalkMetropolis(self._fll, new_state_fn=cauchy_new_state_fn(scale=proposal_scale,
                                                                                           dtype=np.float32))
        key, sample_key = random.split(random.PRNGKey(random_key))

        # Set up the Number of Independent Chains requested:
        initial_values = []
        for index in range(num_chains):
            initial_values.append(initial_value)
        initial_values = np.stack(initial_values)

        if self.lower_bounds is None or self.upper_bounds is None:
            print("Drawing from the Markov Chain...")
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_values,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, results: results.target_log_prob,
                                                        num_burnin_steps=burn_in,
                                                        seed=key,
                                                        parallel_iterations=self.number_of_cores)
            print("---------------------------------")
            states = np.concatenate(states)
            return states, is_accepted
        elif self.lower_bounds is not None and self.upper_bounds is not None:
            print("Drawing from the Markov Chain...")
            states, is_accepted = tfp.mcmc.sample_chain(num_iters,
                                                        current_state=initial_values,
                                                        kernel=kernel,
                                                        trace_fn=lambda _, pkr: pkr.log_accept_ratio,
                                                        num_burnin_steps=burn_in,
                                                        seed=key,
                                                        parallel_iterations=self.number_of_cores)
            new_states, log_jacobian = transform_parameters(states, self.lower_bounds, self.upper_bounds)
            print("---------------------------------")
            new_states = np.concatenate(new_states)
            return new_states, is_accepted
        else:
            raise ValueError("Please make sure your parameter bounds are properly formatted.  "
                             "At a given index, both lower and upper bounds must be equal and must be"
                             "either float type or None.")


@jit
def transform_parameters(states, lower_bounds, upper_bounds):
    """
    After drawing values from the MCMC chain that transforms to be unconstrained, transform back to the constrained
    parameter space.  Also calculates the jacobian of this transform.
    Args:
        upper_bounds: Array-like floats.  The user-given upper bounds on the parameter space.
        lower_bounds: Array-like floats.  The user-given lower bounds on the parameter space.
        states: Matrix. The unconstrained states.

    Returns:
        c_states: the constrained states.
        transform_log_jacobian: the log of the jacobian of the logit or log transform, depending on which was used.
    """
    # TensorFlow follows a [num_samples, num_chains, parameter_dimension] format.
    # In certain cases, we need to reformat the given numpy array.
    if len(states.shape) < 3:
        # Handle the case where we are transforming a single parameter vector
        if len(states.shape) == 1:
            num_chains = 1
            num_samples = 1
            parameter_dimension = states.shape[0]
        else:
            num_chains = states.shape[0]
            num_samples = 1
            parameter_dimension = states.shape[1]
        states = np.array(states).reshape(num_samples, num_chains, parameter_dimension)
    else:
        num_chains = states.shape[1]
        num_samples = states.shape[0]

    new_states = np.zeros([num_samples, num_chains, 1])
    transform_log_jacobian = 0
    if lower_bounds is None and upper_bounds is None:
        return states, transform_log_jacobian
    else:
        for index in range(len(lower_bounds)):
            if lower_bounds[index] is None and upper_bounds[index] is None:
                new_states = np.concatenate((new_states.reshape(states.shape[0], states.shape[1], index + 1),
                                             np.array([states[:, :, index]]).reshape(states.shape[0],
                                                                                     states.shape[1], 1)), axis=2)
            elif lower_bounds[index] is not None and upper_bounds[index] is not None:
                # Perform the logit transform.
                inv_logit_value = np.array(inverse_logit(states[0, index]), float)
                transform_log_jacobian += np.array(np.log((upper_bounds[index] - lower_bounds[index]) *
                                                          inv_logit_value * (1. - inv_logit_value)), float)
                logit_transform_states = np.array([lower_bounds[index] +
                                                   (upper_bounds[index] - lower_bounds[index]) *
                                                   inverse_logit(states[:, :, index])], float)
                new_states = np.concatenate((new_states.reshape(states.shape[0], states.shape[1], index + 1),
                                             logit_transform_states.reshape(states.shape[0],
                                                                            states.shape[1], 1)), axis=2)
            elif upper_bounds[index] is None:
                # Perform the lower logarithmic transform.
                transform_log_jacobian += np.array(states[0, index], float)
                log_transform_states = np.array([np.exp(states[:, :, index]) + lower_bounds[index]], float)
                new_states = np.concatenate((new_states.reshape(states.shape[0], states.shape[1], index + 1),
                                             log_transform_states.reshape(states.shape[0],
                                                                          states.shape[1], 1)), axis=2)
            elif lower_bounds[index] is None:
                # Perform the upper logarithmic transform.
                transform_log_jacobian += np.array(states[0, index], float)
                log_transform_states = np.array([lower_bounds[index] - np.exp(states[:, :, index])], float)
                new_states = np.concatenate((new_states.reshape(states.shape[0], states.shape[1], index + 1),
                                             log_transform_states.reshape(states.shape[0],
                                                                          states.shape[1], 1)), axis=2)
            else:
                raise TypeError("Please make sure your parameter bounds are type float or None.")
    return new_states[:, :, 1:].astype(float), transform_log_jacobian


@jit
def inverse_logit(value):
    """
    Perform the inverse of the logit transform.
    Args:
        value: scalar or array-like parameter values.

    Returns:
        inverse logit: type same as value.  The parameters fed through the inverse logit function.
    """
    return (1. + np.exp(-value)) ** (-1.)
