#!/usr/bin/env python3

"""
    A python file that holds the user-written functions for the Data Generating Algorithm, the Evaluation Function,
    and the Log Likelihood Function.

    Author: Alexander C. Murph
    Date: 2/14/21
"""

import jax.numpy as np
from jax import scipy, grad


def gamma_cdf_term(alpha, data_value):
    return scipy.special.gammainc(alpha, data_value) / np.exp(scipy.special.gammaln(alpha))


def l2_jac_func(theta, data_matrix):
    gamma_cdf_deriv_wrt_alpha = grad(gamma_cdf_term)
    jacobian_matrix = np.array([gamma_cdf_deriv_wrt_alpha(data_matrix[0] * theta[1]) / \
                                (theta[1] * np.exp(log_likelihood([theta[0], 1], data_matrix[0]))),
                                -data_matrix[0] * (1 / theta[1])])

    for index in range(1, data_matrix.shape[0]):
        jac_mat_temp_row = np.array([gamma_cdf_deriv_wrt_alpha(data_matrix[index] * theta[1]) /
                                     (theta[1] * np.exp(log_likelihood([theta[0], 1], data_matrix[index]))),
                                     -data_matrix[index] * (1 / theta[1])])
        jacobian_matrix = np.concatenate(
            (jacobian_matrix, jac_mat_temp_row), axis=0)
    return np.sqrt(np.linalg.det(np.matmul(jacobian_matrix.transpose(), jacobian_matrix)))


# Note: it seems like functions from jax.scipy.stats works a little faster than user-defined functions.
def log_likelihood(theta, data):
    log_sum = 0
    for index in range(data.shape[0]):
        log_sum += - scipy.special.gammaln(theta[0]) + theta[0] * np.log(theta[1]) + \
                   (theta[0] - 1) * np.log(data[i]) - data[index] * theta[1]
    return log_sum
