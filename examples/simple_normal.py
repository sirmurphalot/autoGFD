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
import time
import matplotlib.pyplot as plt

global true_theta
true_theta = [-0.5, 3.2, 1.0, 1.9, 1.1, 2.5]


def run_example():
    # Establish true parameters, data, and initial theta value:
    n = 25
    data_0 = random.multivariate_normal(random.PRNGKey(13), np.asarray(true_theta[0:3]),
                                        np.diag(np.asarray(true_theta[3:])), shape=[n])
    theta_0 = np.asarray([1., 1., 1., 1., 1., 1.])
    lower_bounds = [None, None, None, 0., 0., 0.]
    upper_bounds = [None, None, None, None, None, None]

    # Create the object and perform NUTS:
    t0 = time.time()
    fhmc = FidHMC(log_likelihood, dga_func, eval_func, 6, data_0, lower_bounds, upper_bounds)
    # With bounds:
    # states, log_accept = fhmc.run_NUTS(num_iters=150, burn_in=50, initial_value=theta_0, step_size=15e-2)
    states, log_accept = fhmc.run_RWM(num_iters=150, burn_in=50, initial_value=theta_0, proposal_scale=1e-2)
    # states, log_accept = fhmc.run_HMC(num_iters=150, burn_in=50, initial_value=theta_0, step_size=15e-2)
    t1 = time.time()

    # states, [new_step_size, log_accept] = fhmc.run_HMC(num_iters=1500, burn_in=1500, step_size=1e-2)
    print("---------------------------------")
    print("MCMC draw complete.")
    print("Acceptance Ratio: ", np.exp(np.log(np.mean(np.exp(np.minimum(log_accept, 0.))))))
    # print("Acceptance Ratio: ", np.mean(log_accept))
    # print("Final Step Size: ", new_step_size)
    print("Execultion time: ", t1-t0)
    print("---------------------------------")

    # Save the data:
    my_path = os.path.dirname(os.path.abspath(__file__))
    np.save(my_path + "/data/SimpleNormal_States.npy", states)
    np.save(my_path + "/data/SimpleNormal_AcceptanceRatio.npy",
            np.exp(np.log(np.mean(np.exp(np.minimum(log_accept, 0.))))))
    np.save(my_path + "/data/SimpleNormal_ExecutionTime.npy", np.array(t1-t0, float))


def graph_results():
    # Get Parameter Names
    col_names = []
    for d in range(len(true_theta)):
        col_names.append("theta_" + str(d))

    # Load the data
    my_path = os.path.dirname(os.path.abspath(__file__))
    states = np.load(my_path + "/data/SimpleNormal_States.npy")
    accept_ratio = np.load(my_path + "/data/SimpleNormal_AcceptanceRatio.npy")
    execution_time = np.load(my_path + "/data/SimpleNormal_ExecutionTime.npy")
    # Graph the parameter draws
    # states = states[(np.abs(stats.zscore(states)) < 3.).all(axis=1)]
    temp_sample_df = pd.DataFrame(states)
    sample_df = temp_sample_df.melt()
    g = sns.displot(sample_df, x="value", row="variable", kind="kde", fill=1, color="blue",
                    height=2.5, aspect=3, facet_kws=dict(margin_titles=True), )
    count = 0
    for ax in g.axes.flat:
        ax.axvline(true_theta[count], color="red")
        count += 1
    g.fig.suptitle("Acceptance Ratio: " + str(accept_ratio) + ", Execution Time: " + str(execution_time))
    g.savefig(my_path+'/plots/SimpleNormal_mcmc_samples.png')

    # Graph the log probability
    # plt.figure()
    # plt.plot(log_probs)
    # plt.ylabel('Target Log Prob')
    # plt.xlabel('Iterations of NUTS')
    # plt.savefig(my_path+'/plots/SimpleNormal_mcmc_log_probability.png')
    #
    

run_example()
graph_results()
