#!/usr/bin/env python3

"""
    Unit tests.
    Author: Alexander Murph
    Date: 2/14/21
"""

import unittest
from lib import FidHMC
import jax.numpy as np
import tensorflow_probability as tfp
from jax import random
from jax.scipy import stats
from lib.FidHMC import FidHMC

tfp = tfp.substrates.jax


class TestFHMC(unittest.TestCase):
    def test_fll(self):
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

        def jac_matrix(theta, data):
            return np.asarray([
                [1., 0., 0., (data[0, 0] - theta[0]) / theta[3], 0., 0.],
                [0., 1., 0., 0., (data[0, 1] - theta[1]) / theta[4], 0.],
                [0., 0., 1., 0., 0., (data[0, 2] - theta[2]) / theta[5]],
                [1., 0., 0., (data[1, 0] - theta[0]) / theta[3], 0., 0.],
                [0., 1., 0., 0., (data[1, 1] - theta[1]) / theta[4], 0.],
                [0., 0., 1., 0., 0., (data[1, 2] - theta[2]) / theta[5]]
            ])

        data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray([-0.5, 3.2, 1.0]), np.identity(3), shape=[2])
        fhmc = FidHMC(log_likelihood, dga_func, eval_func, 3, data_0)
        theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])
        fhmc_log_probability = fhmc._fll(theta_0)

        regular_log_probability = log_likelihood(theta_0, data_0)
        regular_log_probability += np.log(np.sqrt(np.linalg.det(np.matmul(
            jac_matrix(theta_0, data_0).transpose(), jac_matrix(theta_0, data_0)))))

        self.assertAlmostEqual(fhmc_log_probability, regular_log_probability)

if __name__ == '__main__':
    # python -m unittest tests/test_FidHMC.py
    unittest.main()

