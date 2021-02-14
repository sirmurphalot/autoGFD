#!/usr/bin/env python3

"""
    Class to produce the jacobian and hessian of an constraint function on the parameter space.

    Author: Alexander C. Murph
    Date: 2/13/21
"""

import jax.numpy as np
from jax import jacrev, hessian


class DifferentiatorG:
    """
        Class that takes in the constraint function g that defines the manifold and encapsulates the Jacobian matrix
        and Hessian matrix calculations at a point on the manifold theta_0.
    """

    def __init__(self, constraint_function=None):
        self.constraint_function = constraint_function
        self.jacobian_g = jacrev(constraint_function)
        self.hessian_g = None

    def process_hessian(self):
        """
        It is not always necessary to calculate the hessian matrix.  To save on computation time, this will only
        be calculated when expressly called.
        Returns:
            None.  Simply updates the Differentiator class.
        """
        self.hessian_g = hessian(self.constraint_function)

    def calculate_jacobian(self, theta_0):
        """
        Method to calculate Jacobian matrix at a point.  Simple wrapper for Jax's jacobian functionality.
        Note that input to constraint_function is assumed to be the same dimension of theta_0.  Take special care to
        observe the output of constraint_function.
        Args:
            constraint_function: python function g:R^theta_dimension -> R^constraint_dimension, where g(theta)=0
                implicitly expresses the manifold.
            theta_0: jac.numpy row array. The point at which we which to take the Jacobian.

        Returns:
            jacobian matrix using jax reverse mode auto-differentiation.  Dimension should be dxt, where d is dimension of
            parameter theta_0 and t is the dimension of the constraint.
        """
        return self.jacobian_g(theta_0)

    def calculate_hessian(self, theta_0):
        """
        Method that calculates the hessian tensor matrix at a point theta_0.
        Args:
            theta_0: The value at which the hessian matrix is to be evaluated

        Returns:
            hessian_g_tensor_array: A tensor array in the form of a dx(d*t) matrix, where d is the dimension of the
                parameter theta_0 and t is the dimension of the constraint.
        """
        hessian_g = self.hessian_g(theta_0)
        constraint_dimension = hessian_g.shape[0]
        hessian_g_tensor_array = hessian_g[0]
        if constraint_dimension > 1:
            for index in range(1, constraint_dimension):
                hessian_g_tensor_array = np.concatenate((hessian_g_tensor_array, hessian_g[index]), axis=1)
        return hessian_g_tensor_array

    def calculate_projection_matrix(self, theta_0):
        """
        Method to calculate the projection matrix onto the null space of nabla(constraint_function).
        Args:
            theta_0: the value at which we calculate the projection matrix.

        Returns:
            projection_matrix: dxd matrix, where d is the dimension of the parameter theta_0.
        """
        jacobian_g = self.jacobian_g(theta_0)
        identity = np.identity(jacobian_g.shape[1])
        projection_matrix = identity - np.matmul(np.matmul(jacobian_g.transpose(), np.linalg.inv(
            np.matmul(jacobian_g, jacobian_g.transpose()))), jacobian_g)
        return projection_matrix
