"""
    A python file that holds the user-written functions for the Data Generating Algorithm, the Evaluation Function,
    and the Log Likelihood Function.

    Author: Alexander C. Murph
    Date: 2/14/21
"""

import jax.numpy as np
from jax.scipy import stats


def dga_func(theta, rand_quantity):
    # TODO: Check that the dimensions on the .solve are correct here.  I think they might not be.
    A_matrix_lower = np.asarray([[0., 0., 0., 0.],
                                 [theta[0], 0., 0., 0.],
                                 [theta[1], theta[2], 0., 0.],
                                 [theta[3], theta[4], theta[5], 0.]])
    I_plus_A_matrix = np.identity(4) + A_matrix_lower - A_matrix_lower.transpose()
    lambda_matrix = np.diag(theta[6:10])
    mu_vector = theta[10:]
    dga_value = mu_vector.transpose() + np.matmul(I_plus_A_matrix.transpose(),
                                                  np.matmul(np.linalg.solve(I_plus_A_matrix,
                                                                            lambda_matrix), rand_quantity.transpose()))
    return dga_value.astype(float)


def eval_func(theta, data_row):
    A_matrix_lower = np.asarray([[0., 0., 0., 0.],
                                 [theta[0], 0., 0., 0.],
                                 [theta[1], theta[2], 0., 0.],
                                 [theta[3], theta[4], theta[5], 0.]])
    I_plus_A_matrix = np.identity(4) + A_matrix_lower - A_matrix_lower.transpose()
    lambda_inv_matrix = np.diag(np.reciprocal(theta[6:10]))
    mu_vector = theta[10:]
    return np.matmul(lambda_inv_matrix, np.matmul(I_plus_A_matrix,
                                                  np.linalg.solve(I_plus_A_matrix.transpose(),
                                                                  data_row.transpose() -
                                                                  mu_vector.transpose()))).astype(float)


def log_likelihood(theta, data):
    A_matrix_lower = np.asarray([[0., 0., 0., 0.],
                                 [theta[0], 0., 0., 0.],
                                 [theta[1], theta[2], 0., 0.],
                                 [theta[3], theta[4], theta[5], 0.]])
    I_plus_A_matrix = np.identity(4) + A_matrix_lower - A_matrix_lower.transpose()
    lambda_2_matrix = np.diag(np.square(theta[6:10]))
    mu_vector = theta[10:]
    sigma = np.matmul(I_plus_A_matrix.transpose(), np.matmul(
        np.linalg.solve(I_plus_A_matrix, lambda_2_matrix),
        np.linalg.solve(I_plus_A_matrix.transpose(), I_plus_A_matrix)
    ))
    log_sum = np.sum(np.log(stats.multivariate_normal.pdf(data, mu_vector, sigma)))
    return np.array(log_sum, float)
