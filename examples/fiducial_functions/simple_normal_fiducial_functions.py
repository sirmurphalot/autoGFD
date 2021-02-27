#!/usr/bin/env python3

"""
    A python file that holds the user-written functions for the Data Generating Algorithm, the Evaluation Function,
    and the Log Likelihood Function.

    Author: Alexander C. Murph
    Date: 2/14/21
"""

import jax.numpy as np
from jax.scipy import stats


def dga_func(theta, rand_quantity):
    return np.asarray([
        theta[0] + theta[3] * rand_quantity[0],
        theta[1] + theta[4] * rand_quantity[1],
        theta[2] + theta[5] * rand_quantity[2]
    ])


def eval_func(theta, data_row):
    return np.asarray([
        (data_row[0] - theta[0]) / theta[3],
        (data_row[1] - theta[1]) / theta[4],
        (data_row[2] - theta[2]) / theta[5]
    ])


def log_likelihood(theta, data):
    log_sum = np.sum(np.log(stats.multivariate_normal.pdf(data, theta[0:3], np.diag(theta[3:]))))
    return log_sum
# def log_likelihood(theta, data):
#     log_sum = 0
#     mu_vector = theta[0:3]
#     cov_matrix = np.diag(theta[3:])
#     inv_cov_matrix = np.diag(np.reciprocal(theta[3:]))
#     for index in range(data.shape[0]):
#         mean_centered = data[index] - mu_vector
#         log_sum += -0.5 * len(data) * np.log(2. * np.pi) - 0.5 * np.log(np.linalg.det(cov_matrix)) - \
#                    0.5 * np.matmul(mean_centered, np.matmul(inv_cov_matrix, mean_centered.transpose()))
#     return log_sum
