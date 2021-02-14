#!/usr/bin/env python3

"""
    A simple python script template.

    Author: Alexander Murph
    Date: $
"""

from fiducialFunctions import *

from FHMC import FHMC
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns


data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray([-0.5, 3.2, 1.0]), np.identity(3), shape=[2])
theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])

# Create the object:
fhmc = FHMC(log_likelihood, dga_func, eval_func, 6, data_0)
states, log_probs = fhmc.run_NUTS(num_iters=100, burn_in=50, initial_value=theta_0)

plt.figure()
plt.plot(log_probs)
plt.ylabel('Target Log Prob')
plt.xlabel('Iterations of NUTS')
plt.show()

print(states)
