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
