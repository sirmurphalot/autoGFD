"""
    A simple wrapper for the user-given Data Generating Algorithm.  As of now, it only makes sure that
    the output is a jax numpy array with float values.

    Author: Alexander C. Murph
    Date: 2/20/21
"""
import jax.numpy as np


class DGAWrapper:

    def __init__(self, dga_function):
        self.dga_func = dga_function

    def get_dga_function(self, parameter_value, random_quantity):
        return np.array(self.dga_func(parameter_value, random_quantity), float)
