"""
    Unit tests.
    Author: Alexander Murph
    Date: $DATE$
"""

import unittest
import jax.numpy as np
from lib.DifferentiatorG import DifferentiatorG


class TestDifferentiation(unittest.TestCase):
    def test_calculate_calculate_jacobian(self):
        def func(theta):
            return np.asarray([
                np.sin(theta[1]) + np.sin(theta[2]),
                np.cos(theta[0])
            ])

        def dfunc(theta):
            return np.asarray([
                [0, np.cos(theta[1]), np.cos(theta[2])],
                [-np.sin(theta[0]), 0, 0]
            ])

        Diff_Class = DifferentiatorG(func)
        theta_0 = np.array([np.pi, 0.5 * np.pi, 2 * np.pi])
        Jf = Diff_Class.calculate_jacobian(theta_0)
        self.assertTrue(np.allclose(Jf, dfunc(theta_0)))

    def test_calculate_hessian(self):
        def func(theta):
            return np.asarray([
                np.exp(3.0 * theta[1]) + np.exp(5.0 * theta[2]),
                np.exp(2.0 * theta[0])
            ])

        def dfunc(theta):
            return np.asarray([
                [0., 0., 0., 4., 0., 0.],
                [0., 9., 0., 0., 0., 0.],
                [0., 0., 25., 0., 0., 0.]
            ])

        Diff_Class = DifferentiatorG(func)
        Diff_Class.process_hessian()
        theta_0 = np.array([0.0, 0.0, 0.0])
        Hf = Diff_Class.calculate_hessian(theta_0)
        self.assertTrue(np.allclose(Hf, dfunc(theta_0)))

    def test_calculate_projection_matrix(self):
        def func(theta):
            return np.asarray([
                theta[0] ** 2. + theta[1] ** 2. - 1.
            ])

        def projection(theta):
            return np.asarray([
                [1. - theta[0] ** 2., -theta[0] * theta[1]],
                [-theta[0] * theta[1], 1. - theta[1] ** 2.]
            ])

        Diff_Class = DifferentiatorG(func)
        theta_0 = np.array([(0.25)**0.5, (0.75)**0.5])
        Pf = Diff_Class.calculate_projection_matrix(theta_0)

        self.assertTrue(np.allclose(Pf, projection(theta_0)))


if __name__ == '__main__':
    # Run unittest.main() in commandline
    # For current file structure, run the following:
    # python -m unittest tests/test_DifferentiatorG.py
    unittest.main()
