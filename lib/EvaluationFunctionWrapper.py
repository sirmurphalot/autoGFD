import jax.numpy as np


class EvaluationFunctionWrapper:
    """
        A simple wrapper for the user-given Evaluation function.  As of now, it only makes sure that
        the output is a jax numpy array with float values.

        Args:
            eval_function: user-given python callable.  Instructions on how to evaluate the Data Generating Algorithm.
    """
    def __init__(self, eval_function):
        self.eval_func = eval_function

    def get_eval_function(self, parameter_value, data_row):
        """
        Evaluates the eval function at a given data row and parameter instance.
        Coerces everything to be a float, which JAX requires.
        Args:
            parameter_value: array-like.  The current parameter values in the MCMC chain.
            data_row: array-like.  One row of the observed data (assumed to be nxp).

        Returns:
            rand_quantities: array-like.  An evaluation of the eval function to be fed into the DGA.
        """
        return np.array(self.eval_func(parameter_value, data_row), float)
