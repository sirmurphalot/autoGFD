import jax.numpy as np


class DGAWrapper:
    """
        A simple wrapper for the user-given Data Generating Algorithm.  As of now, it only makes sure that
        the output is a jax numpy array with float values.

        Args:
            dga_function: user-given python callable. The fiducial Data Generating Algorithm.
    """
    def __init__(self, dga_function):
        self.dga_func = dga_function

    def get_dga_function(self, parameter_value, random_quantity):
        """
        Takes in a parameter value and the random quantity as defined by the evaluation function.
        Args:
            parameter_value: array-like floats.  The current parameter from the MCMC chain.
            random_quantity: array-like floats.  The random quantity as defined by the evaluation function.

        Returns:
            dga evaluation: array-like floats.  Instance of the DGA.
        """
        return np.array(self.dga_func(parameter_value, random_quantity), float)
