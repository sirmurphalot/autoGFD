"""
    Unit tests.
    Author: Alexander Murph
    Date: $Date$
"""

import unittest
import jax.numpy as np
from lib._autodiff import calculate_gradient, calculate_jacobian


class TestDifferentiation(unittest.TestCase):
    def test_calculate_gradient_scalar(self):
        def func(theta):
            return np.sin(theta)
        def dfunc(theta):
            return np.cos(theta)

        theta_0 = np.pi
        Jf = calculate_gradient(func, theta_0)
        self.assertEqual(Jf, dfunc(theta_0))

    def test_calculate_calculate_jacobian(self):
        def func(theta):
            return np.asarray([
                np.sin(theta[1])+np.sin(theta[2]),
                np.cos(theta[0])
            ])

        def dfunc(theta):
            return np.asarray([
                [0, np.cos(theta[1]), np.cos(theta[2])],
                [-np.sin(theta[0]),0,0]
            ])

        theta_0 = np.array([np.pi, 0.5*np.pi, 2*np.pi])
        Jf = calculate_jacobian(func, theta_0)
        self.assertTrue(np.array_equal(Jf, dfunc(theta_0)))


if __name__ == '__main__':
    # Run unittest.main() in commandline
    unittest.main()
