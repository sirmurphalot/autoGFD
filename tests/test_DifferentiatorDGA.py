#!/usr/bin/env python3

"""
    Unit tests.
    Author: Alexander Murph
    Date: 2/14/21
"""

import unittest
import jax.numpy as np
from lib.DifferentiatorDGA import DifferentiatorDGA


class TestDifferentiator_DGA(unittest.TestCase):

    def test_calculate_fiducial_jacobian_matrix(self):
        def dga(theta, rand_quantity):
            return np.asarray([
                theta[0] + theta[3] * rand_quantity[0],
                theta[1] + theta[4] * rand_quantity[1],
                theta[2] + theta[5] * rand_quantity[2]
            ])

        def eval(theta, data_row):
            return np.asarray([
                (data_row[0] - theta[0]) / theta[3],
                (data_row[1] - theta[1]) / theta[4],
                (data_row[2] - theta[2]) / theta[5]
            ])

        def jac_matrix(theta, data):
            return np.asarray([
                [1., 0., 0., (data[0, 0] - theta[0]) / theta[3], 0., 0.],
                [0., 1., 0., 0., (data[0, 1] - theta[1]) / theta[4], 0.],
                [0., 0., 1., 0., 0., (data[0, 2] - theta[2]) / theta[5]],
                [1., 0., 0., (data[1, 0] - theta[0]) / theta[3], 0., 0.],
                [0., 1., 0., 0., (data[1, 1] - theta[1]) / theta[4], 0.],
                [0., 0., 1., 0., 0., (data[1, 2] - theta[2]) / theta[5]]
            ])

        data_0 = np.asarray([[1.5, -1.0, 2.5],
                             [0.7, -1.8, 2.3]])
        theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])
        Diff_DGA = DifferentiatorDGA(dga, eval, 3, data_0)
        fid_jac_mat = Diff_DGA.calculate_fiducial_jacobian_matrix(theta_0)
        print("The JAX Autodiv Jacobian matrix is:")
        print(jac_matrix(theta_0, data_0))
        print("The Jacobian matrix should be:")
        print(fid_jac_mat)

        self.assertTrue(np.allclose(fid_jac_mat, jac_matrix(theta_0, data_0)))

        self.assertAlmostEqual(Diff_DGA.calculate_fiducial_jacobian_quantity_l2(theta_0),
                               np.sqrt(np.linalg.det(np.matmul(
                                   jac_matrix(theta_0, data_0).transpose(), jac_matrix(theta_0, data_0)))))


if __name__ == '__main__':
    # python -m unittest tests/test_DifferentiatorDGA.py
    unittest.main()
