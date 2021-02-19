#!/usr/bin/env python3

"""
    Unit tests.
    Author: Alexander Murph
    Date: 2/14/21
"""
import os
import unittest
from jax import random
from jax.scipy import stats
import jax.numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfp = tfp.substrates.jax


class TestFHMC(unittest.TestCase):
    def test_main(self):
        true_theta = [-0.5, 3.2, 1.0, 1., 1., 1.]
        theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])
        num_iters = 15000
        burn_in = 5000


        # Create the object:
        def fid_log_likelihood(theta):
            data = random.multivariate_normal(random.PRNGKey(13), np.asarray([-0.5, 3.2, 1.0]), np.identity(3), shape=[2])
            jac_matrix = np.asarray([
                [1., 0., 0., (data[0, 0] - theta[0]) / theta[3], 0., 0.],
                [0., 1., 0., 0., (data[0, 1] - theta[1]) / theta[4], 0.],
                [0., 0., 1., 0., 0., (data[0, 2] - theta[2]) / theta[5]],
                [1., 0., 0., (data[1, 0] - theta[0]) / theta[3], 0., 0.],
                [0., 1., 0., 0., (data[1, 1] - theta[1]) / theta[4], 0.],
                [0., 0., 1., 0., 0., (data[1, 2] - theta[2]) / theta[5]]
            ])

            log_sum = np.sum(np.log(stats.multivariate_normal.pdf(data, theta[0:3], np.diag(theta[3:]))))
            log_sum += np.log(np.sqrt(np.linalg.det(np.matmul(
                jac_matrix.transpose(), jac_matrix))))
            return log_sum


        kernel = tfp.mcmc.NoUTurnSampler(fid_log_likelihood, step_size=1e-3)
        key, sample_key = random.split(random.PRNGKey(13))

        states, log_probs = tfp.mcmc.sample_chain(num_iters,
                                                  current_state=theta_0,
                                                  kernel=kernel,
                                                  trace_fn=lambda _, results: results.target_log_prob,
                                                  num_burnin_steps=burn_in,
                                                  seed=key)

        # Graph the results
        col_names = []
        for d in range(len(theta_0)):
            col_names.append("theta_" + str(d))
        temp_sample_df = pd.DataFrame(states, columns=col_names)
        sample_df = temp_sample_df.melt()

        g = sns.displot(sample_df, x="value", row="variable", kind="kde", fill=1, color="blue",
                        height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )
        count = 0
        for ax in g.axes.flat:
            ax.axvline(true_theta[count], color="red")
            count += 1

        my_path = os.path.dirname(os.path.abspath(__file__))
        g.savefig(my_path+'/testing_withoutJAX/mcmc_samples.png')

        plt.figure()
        plt.plot(log_probs)
        plt.ylabel('Target Log Prob')
        plt.xlabel('Iterations of NUTS')
        plt.savefig(my_path+'/testing_withoutJAX/mcmc_log_probability.png')

        self.assertTrue(True)

if __name__ == '__main__':
    # python -m unittest tests/test_main_withoutJAX.py
    unittest.main()

