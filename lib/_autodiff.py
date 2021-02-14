#!/usr/bin/env python3

"""
    Methods to produce the gradient and hessian of an constraint function on the parameter space.

    Author: Alexander C. Murph
    Date: 2/13/21
"""

import os
import sys
import argparse
import jax.numpy as np
from jax import grad, jacrev, hessian

def calculate_jacobian(constraint_function, theta_0):
    """
    Method that takes in the constraint function g that defines the manifold and outputs the Jacobian matrix at theta_0.
    Note that input to constraint_function is assumed to be the same dimension of theta_0.  Take special care to
    observe the output of constraint_function.
    Args:
        constraint_function: python function g:R^theta_dimension -> R^constraint_dimension, where g(theta)=0
            implicitly expresses the manifold.
        theta_0: jac.numpy row array. The point at which we which to take the Jacobian.

    Returns:
        jacobian matrix using jax reverse mode auto-differentiation
    """
    jacobian_g = jacrev(constraint_function)
    return jacobian_g(theta_0)

def calculate_hessian(constraint_function, theta_0):

    return hessian_matrix

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))