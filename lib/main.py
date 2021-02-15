#!/usr/bin/env python3

"""
    Example of using the fiducial NUTS sampler.

    Author: Alexander Murph
    Date: 2/14/21
"""

from fiducialFunctions import *
import jax.numpy as np
from FHMC import FHMC
from jax import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Establish true parameters, data, and initial theta value
true_theta = [-0.5, 3.2, 1.0, 1., 1., 1.]
data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray([-0.5, 3.2, 1.0]), np.identity(3), shape=[2])
theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])

# Create the object and perform NUTS:
fhmc = FHMC(log_likelihood, dga_func, eval_func, 6, data_0)
states, log_probs = fhmc.run_NUTS(num_iters=50, burn_in=25, initial_value=theta_0)

# Graph the results
col_names = []
for d in range(len(theta_0)):
    col_names.append("theta_" + str(d))
temp_sample_df = pd.DataFrame(states, columns=col_names)
sample_df = temp_sample_df.melt()

g = sns.displot(sample_df, x="value", row="variable", kind="kde", fill=1, color = "blue",
                height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )
count = 0
for ax in g.axes.flat:
    ax.axvline(true_theta[count], color="red")
    count += 1

g.savefig('plots/mcmc_samples.png')

plt.figure()
plt.plot(log_probs)
plt.ylabel('Target Log Prob')
plt.xlabel('Iterations of NUTS')
plt.savefig('plots/mcmc_log_probability.png')
