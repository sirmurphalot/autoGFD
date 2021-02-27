from tensorflow_probability.substrates import jax as tfp
from jax import ops, jit
import jax.numpy as np


class LogLikelihoodWrapper:
    """
        A wrapper class for the user-given log likelihood function.  Implements the bijector and adds in a jacobian
        value if bounds are given on the parameter space.

        Author: Alexander C. Murph
        Date: 2/20/21
    """

    def __init__(self, raw_likelihood, data, lower_bounds, upper_bounds):
        self.ll = raw_likelihood
        self.data = data
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def get_log_likelihood(self, parameter_vector):
        """
        Gets the log likelihood from the wrapper class.  Implements parameter transformations as necessary.
        Sets the fixed data values.  Forces the output to be a float.
        Args:
            parameter_vector: row vector numpy array, the parameter values.

        Returns:
            log likelihood: scalar.
        """
        if self.lower_bounds is None and self.upper_bounds is None:
            log_likelihood_value_raw = self.ll(parameter_vector, self.data)
            log_likelihood_value = np.array(log_likelihood_value_raw, float)
            return log_likelihood_value
        elif self.lower_bounds is None:
            raise ValueError("If lower bounds of parameters are set, upper bounds must be as well.")
        elif self.upper_bounds is None:
            raise ValueError("If upper bounds of parameters are set, lower bounds must be as well.")
        else:
            assert len(parameter_vector) == len(self.lower_bounds), "Please make sure the dimension of your bounds " \
                                                                    "matches your input vector. "
            assert len(parameter_vector) == len(self.upper_bounds), "Please make sure the dimension of your bounds " \
                                                                    "matches your input vector. "
            # Transform the parameters to be in the bounds, calculate the jacobian term:
            transformed_parameter_vector = parameter_vector
            transform_log_jacobian = 0
            for index in range(len(parameter_vector)):
                if self.lower_bounds[index] is None and self.upper_bounds[index] is None:
                    continue
                elif type(self.lower_bounds[index]) is float and type(self.upper_bounds[index]) is float:
                    # Perform the logit transform and add in the appropriate log jacobian.
                    inv_logit_value = np.array(inverse_logit(parameter_vector[index]), float)
                    transform_log_jacobian += np.array(np.log((self.upper_bounds[index] - self.lower_bounds[index]) *
                                                              inv_logit_value * (1. - inv_logit_value)), float)
                    likelihood_parameter_value = self.lower_bounds[index] + \
                                                 (self.upper_bounds[index] - self.lower_bounds[index]) * inv_logit_value
                    transformed_parameter_vector = ops.index_update(transformed_parameter_vector,
                                                                    ops.index[index], likelihood_parameter_value)
                elif type(self.lower_bounds[index]) is float and self.upper_bounds[index] is None:
                    # Perform the lower logarithmic transform and add in the appropriate log jacobian.
                    exp_value = np.array(np.exp(parameter_vector[index]), float)
                    transform_log_jacobian += np.array(parameter_vector[index], float)
                    likelihood_parameter_value = self.lower_bounds[index] + exp_value
                    transformed_parameter_vector = ops.index_update(transformed_parameter_vector,
                                                                    ops.index[index], likelihood_parameter_value)
                elif self.lower_bounds[index] is None and type(self.upper_bounds[index]) is float:
                    # Perform the upper logarithmic transform and add in the appropriate log jacobian.
                    exp_value = np.array(np.exp(parameter_vector[index]), float)
                    transform_log_jacobian += np.array(parameter_vector[index], float)
                    likelihood_parameter_value = self.upper_bounds[index] - exp_value
                    transformed_parameter_vector = ops.index_update(transformed_parameter_vector,
                                                                    ops.index[index], likelihood_parameter_value)
                else:
                    raise TypeError("Please make sure your parameter bounds are type float or None.")
            # Get the base likelihood, add the jacobian, make sure it is a float:
            likelihood_value_raw = self.ll(transformed_parameter_vector, self.data)
            likelihood_value_raw += transform_log_jacobian
            likelihood_value = np.array(likelihood_value_raw, float)
            return transformed_parameter_vector, likelihood_value

@jit
def inverse_logit(value):
    return (1. + np.exp(-value)) ** (-1.)
