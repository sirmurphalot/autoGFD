"""
    Example of using the fiducial NUTS sampler.
    Note --> for venv mishaps, try $hash -r
    Author: Alexander Murph
    Date: 2/14/21
"""
import os
from examples.fiducial_functions.simple_normal_fiducial_functions import *
import math
import jax.numpy as np
from lib.FidHMC import FidHMC
from jax import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

global true_theta
true_theta = [-0.5, 3.2, 1.0, 1., 1., 1.]


def run_example():
    # Establish true parameters, data, and initial theta value:
    data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray(true_theta[0:3]),
                                        np.diag(np.asarray(true_theta[3:])), shape=[2])
    theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])

    # Create the object and perform NUTS:
    fhmc = FidHMC(log_likelihood, dga_func, eval_func, 6, data_0)
    states, log_probs = fhmc.run_NUTS(num_iters=15000, burn_in=5000, initial_value=theta_0)

    # Save the data:
    my_path = os.path.dirname(os.path.abspath(__file__))
    np.save(my_path + "/data/SimpleNormal_States.npy", states)
    np.save(my_path + "/data/SimpleNormal_LogProbs.npy", log_probs)


def graph_results():
    # Get Parameter Names
    col_names = []
    for d in range(len(true_theta)):
        col_names.append("theta_" + str(d))

    # Load the data
    my_path = os.path.dirname(os.path.abspath(__file__))
    states = np.load(my_path + "/data/SimpleNormal_States.npy")
    log_probs = np.load(my_path + "/data/SimpleNormal_LogProbs.npy")

    # Print out the acceptance probability:
    # acceptance_prob = math.exp(np.min(log_accept_ratio, 0.))

    # Graph the parameter draws
    temp_sample_df = pd.DataFrame(states, columns=col_names)
    sample_df = temp_sample_df.melt()
    g = sns.displot(sample_df, x="value", row="variable", kind="kde", fill=1, color = "blue",
                    height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )
    count = 0
    for ax in g.axes.flat:
        ax.axvline(true_theta[count], color="red")
        count += 1
    g.savefig(my_path+'/plots/SimpleNormal_mcmc_samples.png')

    # Graph the log probability
    plt.figure()
    plt.plot(log_probs)
    plt.ylabel('Target Log Prob')
    plt.xlabel('Iterations of NUTS')
    plt.savefig(my_path+'/plots/SimpleNormal_mcmc_log_probability.png')


run_example()
graph_results()