#!/usr/bin/env python3

"""
    Class to produce the jacobian matrix of a given Data Generating Algorithm.

    Author: Alexander C. Murph
    Date: 2/14/21
"""
from jax import jacrev, jit
import jax.numpy as np


class DifferentiatorDGA:
    def __init__(self, dga_function, evaluation_function, parameter_dimension, observed_data):
        self.dga_function = dga_function
        try:
            self.jacobian_dga = jit(jacrev(dga_function))
        except:
            print("DGA function not properly constructed.")
        self.evaluation_function = evaluation_function
        self.param_dim = parameter_dimension
        self.data = observed_data
        # Note that data is assumed to be nxp, where n is the number of observed iid samples.
        self.num_samples = observed_data.shape[0]

    def calculate_fiducial_jacobian_matrix(self, theta_0):
        """
        Method to get the full jacobian matrix based on the DGA and Evaluation functions.
        Args:
            theta_0: A point at which we wish to calculate the full fiducial jacobian matrix.

        Returns:
            fiducial_jacobian_matrix: full fiducial jacobian matrix.
        """
        first_eval = self.evaluation_function(theta_0, self.data[0])
        fiducial_jacobian_matrix = self.jacobian_dga(theta_0, first_eval)
        if self.num_samples > 1:
            for index in range(1, self.num_samples):
                temp_eval = self.evaluation_function(theta_0, self.data[index])
                fiducial_jacobian_matrix = np.concatenate(
                    (fiducial_jacobian_matrix, self.jacobian_dga(theta_0, temp_eval)), axis=0)
        return fiducial_jacobian_matrix.astype(float)

    def calculate_fiducial_jacobian_quantity_l2(self, theta_0):
        """
        Method to get the jacobian determinant based on the DGA and Evaluation functions.
        Args:
            theta_0: A point at which we wish to calculate the fiducial jacobian determinant based on the
                l2 norm.

        Returns:
            fid_jac: the fiducial jacobian determinant based on the l2 norm.
        """
        fid_jac_matrix = self.calculate_fiducial_jacobian_matrix(theta_0)
        fid_jac = matrix_inner_product_function(fid_jac_matrix)
        return fid_jac


@jit
def matrix_inner_product_function(fid_jac_matrix):
    return np.sqrt(np.linalg.det(np.matmul(fid_jac_matrix.transpose(), fid_jac_matrix)))