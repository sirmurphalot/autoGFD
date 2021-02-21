from tensorflow_probability.substrates import jax as tfp
from jax import ops
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
            print("If lower bounds of parameters are set, upper bounds must be as well.")
            raise ValueError
        elif self.upper_bounds is None:
            print("If upper bounds of parameters are set, lower bounds must be as well.")
            raise ValueError
        else:
            assert len(parameter_vector) == len(self.lower_bounds), "Please make sure the dimension of your bounds " \
                                                                    "matches your input vector. "
            assert len(parameter_vector) == len(self.upper_bounds), "Please make sure the dimension of your bounds " \
                                                                    "matches your input vector. "
            # Transform the parameters to be in the bounds, calculate the jacobian term:
            transformed_parameter_vector = parameter_vector
            transform_log_jacobian = 0
            for index in range(len(parameter_vector)):
                if self.lower_bounds[index] is None or self.upper_bounds[index] is None:
                    continue
                else:
                    bijector = tfp.bijectors.SoftClip(low=self.lower_bounds[index],
                                                      high=self.upper_bounds[index], hinge_softness=5)
                    new_value = bijector.forward([parameter_vector[index]])
                    print(new_value)
                    print([parameter_vector[index]])
                    transform_log_jacobian += bijector.forward_log_det_jacobian([parameter_vector[index]], 1)
                    transformed_parameter_vector = ops.index_update(transformed_parameter_vector,
                                                                    ops.index[index], new_value[0])
            # Get the base likelihood, add the jacobian, make sure it is a float:
            likelihood_value_raw = self.ll(transformed_parameter_vector, self.data)
            likelihood_value_raw += transform_log_jacobian
            likelihood_value = np.array(likelihood_value_raw, float)
            return likelihood_value
