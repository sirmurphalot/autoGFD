"""
    A simple wrapper for the user-given Evaluation function.  As of now, it only makes sure that
    the output is a jax numpy array with float values.

    Author: Alexander C. Murph
    Date: 2/20/21
"""
import jax.numpy as np


class EvaluationFunctionWrapper:

    def __init__(self, eval_function):
        self.eval_func = eval_function

    def get_eval_function(self, parameter_value, data_row):
        return np.array(self.eval_func(parameter_value, data_row), float)
